from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np
import heapq
from models.losses import FocalLoss
from models.losses import RegL1Loss, RegLoss, NormRegL1Loss, RegWeightedL1Loss
from models.utils import _sigmoid
from utils.oracle_utils import gen_oracle_map
from models.data_parallel import DataParallel

class CtdetLoss(torch.nn.Module):
    def __init__(self, opt):
        super(CtdetLoss, self).__init__()
        print("opt.mse_loss  is", opt.mse_loss )
        self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
        self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
            RegLoss() if opt.reg_loss == 'sl1' else None
        self.crit_wh = torch.nn.L1Loss(reduction='sum') if opt.dense_wh else \
            NormRegL1Loss() if opt.norm_wh else \
                RegWeightedL1Loss() if opt.cat_spec_wh else self.crit_reg
        self.opt = opt

    def forward(self, outputs, batch):
        opt = self.opt
        hm_loss, wh_loss, off_loss = 0, 0, 0
        for s in range(opt.num_stacks):
            output = outputs[s]
            if not opt.mse_loss:
                output['hm'] = _sigmoid(output['hm'])

            if opt.eval_oracle_hm:
                output['hm'] = batch['hm']
            if opt.eval_oracle_wh:
                output['wh'] = torch.from_numpy(gen_oracle_map(
                    batch['wh'].detach().cpu().numpy(),
                    batch['ind'].detach().cpu().numpy(),
                    output['wh'].shape[3], output['wh'].shape[2])).to(opt.device)
            if opt.eval_oracle_offset:
                output['reg'] = torch.from_numpy(gen_oracle_map(
                    batch['reg'].detach().cpu().numpy(),
                    batch['ind'].detach().cpu().numpy(),
                    output['reg'].shape[3], output['reg'].shape[2])).to(opt.device)

            hm_loss += self.crit(output['hm'], batch['hm']) / opt.num_stacks
            if opt.wh_weight > 0:
                if opt.dense_wh:
                    mask_weight = batch['dense_wh_mask'].sum() + 1e-4
                    wh_loss += (
                                       self.crit_wh(output['wh'] * batch['dense_wh_mask'],
                                                    batch['dense_wh'] * batch['dense_wh_mask']) /
                                       mask_weight) / opt.num_stacks
                elif opt.cat_spec_wh:
                    wh_loss += self.crit_wh(
                        output['wh'], batch['cat_spec_mask'],
                        batch['ind'], batch['cat_spec_wh']) / opt.num_stacks
                else:
                    wh_loss += self.crit_reg(
                        output['wh'], batch['reg_mask'],
                        batch['ind'], batch['wh']) / opt.num_stacks

            if opt.reg_offset and opt.off_weight > 0:
                off_loss += self.crit_reg(output['reg'], batch['reg_mask'],
                                          batch['ind'], batch['reg']) / opt.num_stacks

        loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + \
               opt.off_weight * off_loss
        loss_stats = {'loss': loss, 'hm_loss': hm_loss,
                      'wh_loss': wh_loss, 'off_loss': off_loss}
        return loss, loss_stats


class ModelWithLoss(torch.nn.Module):
    def __init__(self, model, loss):
        super(ModelWithLoss, self).__init__()
        self.model = model
        self.loss = loss

    def forward(self, batch):
        outputs = self.model(batch['input'])
        loss, loss_stats = self.loss(outputs, batch)
        return outputs[-1], loss, loss_stats

class CtTrainer(object):
    def __init__(self, opt,  model, optimizer=None,  dataloader=None):
        # super(CtTrainer, self).__init__( opt, model, optimizer=optimizer, dataloader=dataloader)
        self.opt = opt
        self.model = model
        # self.criterion = criterion
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.iterations = 0
        self.loss_stats, self.loss = self._get_losses(opt)
        self.model_with_loss = ModelWithLoss(model, self.loss)
        self.num_iters= len(self.dataloader)
        self.stats = {}
        self.phase = "train"

        self.plugin_queues = {
            'iteration': [],
            'epoch': [],
            'batch': [],
            'update': [],
        }

    def _get_losses(self, opt):
        loss_states = ['loss', 'hm_loss', 'wh_loss', 'off_loss']
        loss = CtdetLoss(opt)
        return loss_states, loss

    def set_device(self, gpus, chunk_sizes, device):
        if len(gpus) > 1:
            self.model_with_loss = DataParallel(
                self.model_with_loss, device_ids=gpus,
                chunk_sizes=chunk_sizes).to(device)
        else:
            self.model_with_loss = self.model_with_loss.to(device)

        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device=device, non_blocking=True)

    def register_plugin(self, plugin):
        # 注册插件
        plugin.register(self)

        # 插件的触发间隔,一般是这样的形式[(1, 'iteration'), (1, 'epoch')]
        intervals = plugin.trigger_interval

        if not isinstance(intervals, list):
            intervals = [intervals]
        for duration, unit in intervals:
            # unit 是事件的触发类别
            queue = self.plugin_queues[unit]
            '''添加事件， 这里的duration就是触发间隔,，以后在调用插件的时候，
            会进行更新  duration 决定了比如在第几个iteration or epoch 触发事件。len(queue)这里应当理解为优先级（越小越高）
            【在相同duration的情况下决定调用的顺序】，根据加入队列的早晚决定。'''
            queue.append((duration, len(queue), plugin))

    def call_plugins(self, queue_name, time, *args):
        # 调用插件
        args = (time,) + args
        # 这里的time 最基本的意思是次数,如(iteration or epoch)
        queue = self.plugin_queues[queue_name]
        if len(queue) == 0:
            return
        while queue[0][0] <= time:
            '''如果队列第一个事件的duration（也就是触发时间点）小于当前times'''
            plugin = queue[0][2]
            '''调用相关队列相应的方法，所以如果是继承Plugin类的插件，
                       必须实现 iteration、batch、epoch和update中的至少一个且名字必须一致。'''
            getattr(plugin, queue_name)(*args)
            for trigger in plugin.trigger_interval:
                if trigger[1] == queue_name:
                    interval = trigger[0]
            '''根据插件的事件触发间隔，来更新事件队列里的事件 duration'''
            new_item = (time + interval, queue[0][1], plugin)
            heapq.heappushpop(queue, new_item)
            '''加入新的事件并弹出最小堆的堆头。最小堆重新排序。'''

    def run(self, epochs=1):
        for q in self.plugin_queues.values():
            '''对四个事件调用序列进行最小堆排序。'''
            heapq.heapify(q)

        for i in range(1, epochs + 1):
            self.train()
            # 进行每次epoch 的更新
            self.call_plugins('epoch', i)

    def train(self):
        model_with_loss = self.model_with_loss
        if self.phase == 'train':
            model_with_loss.train()
        for iter_id, batch in enumerate(self.dataloader):
            batch_input = batch['input']
            batch_hm = batch['hm']
            print("batch hm is: ", batch_hm)
            batch_reg_mask, batch_ind, batch_wh, batch_reg = batch['reg_mask'], batch['ind'], batch['wh'], batch['reg']
            # self.call_plugins('batch', iter_id, batch_input, batch_hm,  batch_reg_mask, batch_ind, batch_wh, batch_reg)
            self.call_plugins('batch', iter_id, batch_input, batch_hm)
            if iter_id >= self.num_iters:
                break
            # data_time.update(time.time() - end)
            # plugin_data = [None, None, None, None]
            plugin_data = [None, None]
            for k in batch:
                if k != 'meta':
                    batch[k] = batch[k].to(device=self.opt.device, non_blocking=True)
            # def closure():
            batch_output, loss, loss_stats = self.model_with_loss(batch)
            # print("batch_output is", batch_output)
            # batch_output = self.model(input_var)
            loss = loss.mean()
            print('loss is,', loss)

            if plugin_data[0] is None:
                plugin_data[0] = batch_output['hm'].data
                # plugin_data[1] = batch_output['wh'].data
                # plugin_data[2] = batch_output['reg'].data
                # plugin_data[2] = batch_output['reg'].data
                plugin_data[1] = loss.data
                # return loss
            if self.phase == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            print("current loss is: ", loss)
            self.call_plugins('iteration', iter_id, batch_input, batch_hm
                              *plugin_data)
            # self.call_plugins('iteration', iter_id, batch_input, batch_hm,  batch_reg_mask, batch_ind, batch_wh, batch_reg,
            #                   *plugin_data)
            self.call_plugins('update', iter_id, self.model)
        # for i, data in enumerate(self.dataloader, self.iterations + 1):
        #     batch_input, batch_target = data
        #     # 在每次获取batch data 后进行更新
        #     self.call_plugins('batch', i, batch_input, batch_target)
        #     input_var = batch_input
        #     target_var = batch_target
        #     # 这里是给后续插件做缓存部分数据,这里是网络输出与loss
        #     plugin_data = [None, None]
        #
        #     def closure():
        #         batch_output = self.model(input_var)
        #         loss = self.criterion(batch_output, target_var)
        #         loss.backward()
        #         if plugin_data[0] is None:
        #             plugin_data[0] = batch_output.data
        #             plugin_data[1] = loss.data
        #         return loss
        #
        #     self.optimizer.zero_grad()
        #     self.optimizer.step(closure)
        #     self.call_plugins('iteration', i, batch_input, batch_target,
        #                       *plugin_data)
        #     self.call_plugins('update', i, self.model)

        self.iterations += i



class Plugin(object):
    def __init__(self, interval=None):
        if interval is None:
            interval = []
        self.trigger_interval = interval

    def register(self, trainer):
        raise NotImplementedError

from .monitor import Monitor

class LossMonitor(Monitor):
    stat_name = 'loss'
    #该插件的作用为简单记录每次的loss
    def _get_value(self, iteration, input, target, output, loss):
        return loss.item()

from collections import defaultdict
# from .plugin import Plugin


class Logger_Trainer(Plugin):
    alignment = 4
    #不同字段之间的分隔符
    separator = '#' * 80

    def __init__(self, fields, interval=None):
        if interval is None:
            interval = [(1, 'iteration'), (1, 'epoch')]
        super(Logger_Trainer, self).__init__(interval)

        #需要打印的字段,如loss acc
        self.field_widths = defaultdict(lambda: defaultdict(int))
        self.fields = list(map(lambda f: f.split('.'), fields))
        # 遵循XPath路径的格式。以AccuracyMonitor为例子，如果你想打印所有的状态，
        # 那么你只需要令fields=[AccuracyMonitor.stat_name]，也就是，['accuracy']，
        # 而如果你想只打印AccuracyMonitor的子状态'last'，那么你就只需要设置为
        # ['accuracy.last'],而这里的split当然就是为了获得[['accuracy', 'last']]
        # 这是为了之后的子状态解析（类似XPath路径解析）所使用的。

    def _join_results(self, results):
        # 这个函数主要是将获得的子状态的结果进行组装。
        joined_out = map(lambda i: (i[0], ' '.join(i[1])), results)
        joined_fields = map(lambda i: '{}: {}'.format(i[0], i[1]), joined_out)
        return '\t'.join(joined_fields)

    def log(self, msg):
        print(msg)

    def register(self, trainer):
        self.trainer = trainer

    def gather_stats(self):
        result = {}
        return result

    def _align_output(self, field_idx, output):
        #对其输出格式
        for output_idx, o in enumerate(output):
            if len(o) < self.field_widths[field_idx][output_idx]:
                num_spaces = self.field_widths[field_idx][output_idx] - len(o)
                output[output_idx] += ' ' * num_spaces
            else:
                self.field_widths[field_idx][output_idx] = len(o)

    def _gather_outputs(self, field, log_fields, stat_parent, stat, require_dict=False):
        # 这个函数是核心，负责将查找到的最底层的子模块的结果提取出来。
        output = []
        name = ''
        if isinstance(stat, dict):
            '''
            通过插件的子stat去拿到每一轮的信息,如LOSS等
            '''
            log_fields = stat.get(log_fields, [])
            name = stat.get('log_name', '.'.join(field))
            # 找到自定义的输出名称。y有时候我们并不像打印对应的Key出来，所以可以
            # 在写插件的时候增加多一个'log_name'的键值对，指定打印的名称。默认为
            # field的完整名字。传入的fileds为['accuracy.last']
            # 那么经过初始化之后，fileds=[['accuracy',
            # 'last']]。所以这里的'.'.join(fields)其实是'accuracy.last'。
            # 起到一个还原名称的作用。
            for f in log_fields:
                output.append(f.format(**stat))
        elif not require_dict:
            # 在这里的话，如果子模块stat不是字典且require_dict=False
            # 那么他就会以父模块的打印格式和打印单位作为输出结果的方式。
            name = '.'.join(field)
            number_format = stat_parent.get('log_format', '')
            unit = stat_parent.get('log_unit', '')
            fmt = '{' + number_format + '}' + unit
            output.append(fmt.format(stat))
        return name, output

    def _log_all(self, log_fields, prefix=None, suffix=None, require_dict=False):
        results = []
        for field_idx, field in enumerate(self.fields):
            parent, stat = None, self.trainer.stats
            for f in field:
                parent, stat = stat, stat[f]
            name, output = self._gather_outputs(field, log_fields,
                                                parent, stat, require_dict)
            if not output:
                continue
            self._align_output(field_idx, output)
            results.append((name, output))
        if not results:
            return
        output = self._join_results(results)
        loginfo = []

        if prefix is not None:
            loginfo.append(prefix)
            loginfo.append("\t")

        loginfo.append(output)
        if suffix is not None:
            loginfo.append("\t")
            loginfo.append(suffix)
        self.log("".join(loginfo))

    def iteration(self, *args):
        '''
        :param args:   ( i, batch_input, batch_target,*plugin_data) 的元祖
        :return:
        '''
        self._log_all('log_iter_fields',prefix="iteration:{}".format(args[0]))

    def epoch(self, epoch_idx):
        self._log_all('log_epoch_fields',
                      prefix=self.separator + '\nEpoch summary:',
                      suffix=self.separator,
                      require_dict=True)

