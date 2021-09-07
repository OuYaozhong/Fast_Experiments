import torch
import re


class Observer(object):
    def __init__(self, momentum=None):
        self.min = None
        self.max = None
        self.full_scale = None
        self.same_sign = None
        self._momentum = momentum

    def update(self, x):
        assert isinstance(x, torch.Tensor)
        xmin = x.min().item()
        xmax = x.max().item()
        minmax_sign = xmin * xmax
        if minmax_sign > 0:
            self.same_sign = True
        else:
            self.same_sign = False
        if self._momentum is None or self.min is None or self.max is None:
            self.min = xmin
            self.max = xmax
        else:
            self.min = self._momentum * self.min + (1 - self._momentum) * xmin
            self.max = self._momentum * self.max + (1 - self._momentum) * xmax
        self.full_scale = self.max - self.min

    def set_EMA(self, enable=None, momentum=None):
        assert isinstance(enable, bool)
        if enable:
            assert isinstance(momentum, float)
            assert (momentum > 0) & (momentum < 1)
            self._momentum = momentum
        else:
            self._momentum = None

    def __repr__(self):
        self_name = self.__class__.__name__
        return '{}[\n' \
               'min: {}\n' \
               'max: {}\n' \
               'full_scale: {}\n' \
               'same_sign: {}\n' \
               ']'.format(self_name, self.min, self.max, self.full_scale, self.same_sign)


class BaseQuantizer(object):
    valid_scheme_keys = ['Moving', ('MinMax', 'Symmetric'), 'Adaptive'] # 嵌套深度: 1 层

    def __init__(self, scheme='MovingMinMax', params=None, int_range=None, momentum=None, allow_zp_out_of_bound=True):
        # get quantized data, scale, and zero_point
        if params is not None:
            assert isinstance(params, torch.nn.Parameter)
            self.params = params
        else:
            raise ValueError('请传入data')

        self.allow_zp_out_of_bound = allow_zp_out_of_bound
        # below parameters are left to scheme_praser to set
        self.symmetric = None
        self.momentum = None
        self.adaptive = None
        # end
        # 先让scheme_praser解析scheme参数，设置重要的变量
        self.scheme_praser(scheme=scheme, momentum=momentum)

        self.observer = Observer(momentum=self.momentum)

        if int_range is None:
            int_range = [0, 255]
        assert isinstance(int_range, (list, tuple))
        assert len(int_range) == 2
        assert int_range[1] > int_range[0]
        self.int_range = list(int_range)
        # below paramters are left for _adaptive_quantize_config in self.update to set
        self.int_min = None # the minimum integer after quantization
        self.int_max = None # the maximum integer after quantization
        self.float_min = None # the maximum of floating-point in quantization
        self.float_max = None # the minimum of the floating-point in quantization, and outlier will be clamped
        self.int_full_scale = None
        self.float_full_scale = None
        # end

        self.eps = 1e-9
        self.scale = None
        self.data = None
        self.zp = None

        self.update()

    def scheme_praser(self, scheme, momentum):
        # 检验scheme的有效性
        self.scheme_validate(scheme)
        # set momentum
        if 'Moving' in scheme:
            assert isinstance(momentum, float)
            assert (momentum > 0) & (momentum < 1)
            self.momentum = momentum
        else:
            self.momentum = None
        # set symmetric
        self.symmetric = True if 'Symmetric' in scheme else False
        # set adaptive
        self.adaptive = True if 'Adaptive' in scheme else False
        if 'Adaptive' in scheme:
            if 'MinMax' in scheme:
                print('由于启用Adaptive, MinMax将被忽略')
            elif 'Symmetric' in scheme:
                print('由于启用Adaptive, Symmetric将被忽略')
        return

    def scheme_validate(self, scheme):
        assert isinstance(scheme, str)
        # valid_list内的tuple或list类型，视为互为冲突
        # conflict check
        for valid_key in self.__class__.valid_scheme_keys:
            if isinstance(valid_key, (list, tuple)):
                show_up_key = []
                for key in valid_key:
                    if key in scheme:
                        show_up_key.append(key)
                if len(show_up_key) > 1:
                    raise ValueError('输入的scheme有冲突：{}不可同时输入'.format(show_up_key))
        # 检查冗余
        scheme_left = scheme
        for valid_key in self.__class__.valid_scheme_keys:
            if isinstance(valid_key, (list, tuple)):
                for key in valid_key:
                    scheme_left = scheme_left.replace(key, '')
            else:
                scheme_left = scheme_left.replace(valid_key, '')
        if len(scheme_left) > 0:
            raise ValueError('输入的scheme: {}, 含有非法内容'.format(scheme))

    def _update_quantize_config(self):
        xmin, xmax = self.observer.min, self.observer.max
        # 先根据 self.adaptive 设置好 self.symmetric 与 self.int_min & self.int_max
        if self.adaptive:
            if xmin * xmax > 0: # if same sign
                self.symmetric = True
            else:
                self.symmetric = False
        else:
            self.int_min, self.int_max = self.int_range

        # 以下设置与 self.adaptive 参数无关
        # deal with integer's range
        if self.symmetric:
            if (self.int_range[1] - self.int_range[0]) % 2 == 0:
                self.int_min = self.int_range[0]
                self.int_max = self.int_range[1]
            else:
                self.int_min = self.int_range[0]
                self.int_max = self.int_range[1] - 1
            assert (self.int_max - self.int_min) % 2 == 0
        else:
            self.int_min, self.int_max = self.int_range

        # deal with floating-point's range
        if self.symmetric:
            max_abs_float = max(abs(xmin), abs(xmax))
            self.float_max = max_abs_float
            self.float_min = -max_abs_float
        else:
            self.float_max = xmax
            self.float_min = xmin
        self.int_full_scale = self.int_max - self.int_min
        self.float_full_scale = self.float_max - self.float_min
        return

    def update(self):
        self.observer.update(self.params.data)
        # reset the quantization setting based on new observation
        self._update_quantize_config()
        self.data, self.scale, self.zp = self._quantize(self.params.data)
        self.params.data = self.float()
        return

    def _quantize(self, x):
        # 由于其他函数 self._update_quantize_config 设置好了参数，因而，_quantize函数无论什么情况，都按照一样的流程进行量化
        self._check_data(x)
        scale = self.float_full_scale / self.int_full_scale
        scale = max(scale, self.eps)
        zp = self.int_min - round(self.float_min / scale)
        if not self.allow_zp_out_of_bound:
            zp = min(max(zp, self.int_min), self.int_max)
        if (zp < 0) or (zp > 255):
            print('problem')
        data = (x / scale + zp).round().int()
        if self.symmetric:
            assert zp == (self.int_min + self.int_max) // 2
        return data, scale, zp

    def float(self):
        return self.scale * (self.data.to(torch.float) - self.zp)

    def to(self, *args):
        self.params.data.to(*args)

    def _check_data(self, x):
        assert isinstance(x, torch.Tensor)

    def __repr__(self):
        self_name = self.__class__.__name__
        return '{}(\n' \
               '\tdata:\n' \
               '\t\t{}\n' \
               '\tquantized_max: {}\n' \
               '\tquantized_min: {}\n' \
               '\tscale: {}\n' \
               '\tzero-point: {}\n' \
               '\tSymmetric: {}\n' \
               '\tAdaptive: {})'.format(self_name, self.data, self.int_max, self.int_min, self.scale, self.zp, self.symmetric, self.adaptive)


class Quantizer:
    def __init__(self, named_params, quan_bw=8, momentum=0.99, only_weight=True, first_layer_quantize=True,
                 offset_gen_quantize=True, scheme='Moving', allow_zp_out_of_bound=True):
        self.quan_bit_width = quan_bw
        self.quan_range = [0, pow(2, self.quan_bit_width) - 1]
        self.momentum = momentum
        self.allow_zp_out_of_bound = allow_zp_out_of_bound
        self.quantizers_list = []
        self.quantized_names = []
        for n, p in named_params:
            assert isinstance(p, torch.nn.Parameter)
            if (not first_layer_quantize) & bool(re.search("^conv1", n, re.I)):
                continue
            if (not offset_gen_quantize) & bool(re.search('^deconv_layers\.[0-9]+\.conv_offset_mask', n, re.I)):
                continue
            if only_weight & ('weight' in n):
                self.quantizers_list.append(BaseQuantizer(scheme,
                                                          params=p, int_range=self.quan_range, momentum=0.99,
                                                          allow_zp_out_of_bound=self.allow_zp_out_of_bound))
                self.quantized_names.append(n)
                print('[{}] use [{}] quantization scheme'.format(n, scheme))

    def update(self):
        for quanter in self.quantizers_list:
            quanter.update()


def get_correct(x, scheme='MinMax', qmin=0, qmax=255):
    valid_scheme = ['MinMax', 'Symmetric']
    assert scheme in valid_scheme
    xmin = x.min().item()
    xmax = x.max().item()
    if scheme == 'Symmetric':
        abs_max = max(abs(xmin), abs(xmax))
        xmin = -abs_max
        xmax = abs_max
        assert (qmax - qmin) % 2 == 0
    lsb = (xmax - xmin) / (qmax - qmin)
    ruler = torch.linspace(xmin, xmax, qmax - qmin + 1).to(x.device)
    off_val, off_ind = ruler.abs().min(dim=0)
    offset = ruler[off_ind]
    if scheme == 'Symmetric':
        assert offset == 0
    ruler = ruler - offset
    shape = x.shape
    x = x.flatten()
    x_cor = torch.empty(x.shape, dtype=torch.int32, device=x.device)
    for i in range(len(x)):
        diff = (x[i] - ruler).abs()
        assert len(diff.shape) == 1
        min_val, min_ind = diff.min(dim=0)
        x_cor[i] = min_ind
    x_cor = x_cor.reshape(shape)
    return x_cor, ruler


def check_correct(data_before_quan, data_after_quan, qmin=0, qmax=255):
    assert data_before_quan.shape == data_after_quan.shape
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    data_before_quan = data_before_quan.to(device)
    data_after_quan = data_after_quan.to(device)
    xmin = data_before_quan.min().item()
    xmax = data_before_quan.max().item()
    if xmin * xmax > 0:
        scheme = 'Symmetric'
    else:
        scheme = 'MinMax'
    print('Using ' + scheme)
    if scheme == 'Symmetric':
        if (qmax - qmin) % 2 == 1:
            qmax -= 1
    correct, ruler = get_correct(data_before_quan, scheme=scheme, qmin=qmin, qmax=qmax)
    if scheme == 'Symmetric':
        scale = 2 * max(abs(xmax), abs(xmin)) / (qmax - qmin)
    else:
        scale = (xmax - xmin) / (qmax - qmin)
    data_before_quan_scale = data_before_quan / scale
    eq_map = data_after_quan.eq(correct)
    if eq_map.all():
        return True
    else:
        diff = data_after_quan - correct
        diff_valid = (diff.abs() <= 1).all()
        round_ambiguous_map = (data_before_quan_scale[~eq_map] % 0.5 < 0.1) | (
                    data_before_quan_scale[~eq_map] % -0.5 > -0.1)
        round_ambiguous_valid = round_ambiguous_map.all()
        if diff_valid and round_ambiguous_valid:
            return True
        else:
            return False


def get_all_base(cls):
    bases_list = []
    base = cls
    while True:
        base = base.__base__
        if base is None:
            break
        else:
            bases_list.append(base)
    return bases_list


def same_base_check(cls1, cls2):
    bases_list1 = get_all_base(cls1)
    bases_list2 = get_all_base(cls2)
    for base1 in bases_list1:
        if base1 in bases_list2:
            return True
    return False


if __name__ == '__main__':
    momentum = 0.99
    pre_x = torch.randn(1000, 500) * torch.randint(0, 100, (1000, 500)).float()
    moving_max = pre_x.max().item()
    moving_min = pre_x.min().item()
    x = torch.nn.Parameter(data=pre_x.clone(), requires_grad=False)
    xq = BaseQuantizer(scheme='MovingMinMax', params=x, int_range=[0, 127], momentum=momentum)
    print(xq)
    print('[{}]: {}, {}'.format(-1, moving_max == xq.observer.max, moving_min == xq.observer.min))
    for i in range(100):
        pre_x = torch.randn(1000, 500) * torch.randint(0, 100, (1000, 500)).float()
        x.data = pre_x.clone()
        xq.update()
        moving_max = momentum * moving_max + (1 - momentum) * pre_x.max().item()
        moving_min = momentum * moving_min + (1 - momentum) * pre_x.min().item()
        print('[{}]: {}, {}'.format(i, moving_max==xq.observer.max, moving_min==xq.observer.min))

    print('xq moving_max: {}, xq moving_min: {}'.format(xq.observer.max, xq.observer.min))
    print('my moving_max: {}, my moving_min: {}'.format(moving_max, moving_min))


    pass
