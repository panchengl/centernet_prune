from models.trains.ct_trainer import Plugin
class Monitor(Plugin):

    def __init__(self, running_average=True, epoch_average=True, smoothing=0.7,
                 precision=None, number_format=None, unit=''):
        '''
        :param running_average:
        :param epoch_average:
        :param smoothing:
        :param precision:数字输出精度
        :param number_format:  数字输出格式
        :param unit:
        '''
        if precision is None:
            precision = 4
        if number_format is None:
            number_format = '.{}f'.format(precision)
        # 规定了输出格式
        number_format = ':' + number_format
        '''
        在基类 plugin 中,初始化需要传入interval 参数,此处list[(1, 'iteration'), (1, 'epoch')] 
        代表了插件自身实现的的触发time 跟触发时间
        '''
        super(Monitor, self).__init__([(1, 'iteration'), (1, 'epoch')])

        # 是否平滑
        self.smoothing = smoothing
        # 增量计算均值
        self.with_running_average = running_average
        self.with_epoch_average = epoch_average

        # 输出日志的格式
        self.log_format = number_format
        self.log_unit = unit
        self.log_epoch_fields = None
        self.log_iter_fields = ['{last' + number_format + '}' + unit]
        if self.with_running_average:
            self.log_iter_fields += [' ({running_avg' + number_format + '}' + unit + ')']
        if self.with_epoch_average:
            self.log_epoch_fields = ['{epoch_mean' + number_format + '}' + unit]

    def register(self, trainer):
        self.trainer = trainer
        # 在此处注册的时候,给train 的stats 注册当前状态,比如log 的格式等
        stats = self.trainer.stats.setdefault(self.stat_name, {})
        stats['log_format'] = self.log_format
        stats['log_unit'] = self.log_unit
        stats['log_iter_fields'] = self.log_iter_fields
        if self.with_epoch_average:
            stats['log_epoch_fields'] = self.log_epoch_fields
        if self.with_epoch_average:
            stats['epoch_stats'] = (0, 0)

    def iteration(self, *args):
        # 每个iteration 进行的操作
        stats = self.trainer.stats.setdefault(self.stat_name, {})
        # 通过_get_value 方法拿到每个插件的值,放入到stats中
        stats['last'] = self._get_value(*args)

        if self.with_epoch_average:
            stats['epoch_stats'] = tuple(sum(t) for t in
                                         zip(stats['epoch_stats'], (stats['last'], 1)))

        if self.with_running_average:
            previous_avg = stats.get('running_avg', 0)
            stats['running_avg'] = previous_avg * self.smoothing + \
                                   stats['last'] * (1 - self.smoothing)

    def epoch(self, idx):
        # 每个epoch 进行的操作
        stats = self.trainer.stats.setdefault(self.stat_name, {})
        if self.with_epoch_average:
            # 如果需要计算每轮epoch 的精度等,需要 总数/轮数
            epoch_stats = stats['epoch_stats']
            stats['epoch_mean'] = epoch_stats[0] / epoch_stats[1]
            stats['epoch_stats'] = (0, 0)

