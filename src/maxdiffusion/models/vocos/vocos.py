import jax
import jax.numpy as jnp
from flax import nnx

class ISTFT(nnx.Module):
    """
    使用 JAX segment_sum 进行优化的自定义 ISTFT 实现。
    支持 "same" 和 "center" 填充模式（"center" 待实现）。

    参数:
        n_fft (int): 傅里叶变换的大小。
        hop_length (int): 相邻滑动窗口帧之间的距离。
        win_length (int): 窗口帧和 STFT 滤波器的大小。
        padding (str, optional): 填充类型，"center" 或 "same"。默认为 "same"。
        epsilon (float, optional): 用于归一化时避免除以零的小值。默认为 1e-11。
    """

    def __init__(self, n_fft: int = 1024, hop_length: int = 256, win_length: int = 1024,
                 padding: str = "same", epsilon: float = 1e-11, *, rngs: nnx.Rngs | None = None):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.padding = padding
        self.epsilon = epsilon
        if self.padding not in ["center", "same"]:
            raise ValueError("Padding must be 'center' or 'same'.")
        if self.win_length > self.n_fft:
            raise ValueError(f"win_length ({self.win_length}) cannot be larger than n_fft ({self.n_fft})")
        # 学习型窗口参数
        #self.window = nnx.Param(jnp.hanning(self.win_length))

    def __call__(self, spec):
        """
        执行逆短时傅里叶变换。

        参数:
            spec (jax.Array): 输入频谱图，形状为 (B, N_freq, T)，
                              其中 B=批次大小, N_freq=频率箱数量 (应为 n_fft//2 + 1), T=时间帧数。

        返回:
            jax.Array: 重建的时域信号，形状为 (B, L_out)，
                       其中 L_out 是重建信号的长度。
        """
        # 输入张量维度检查
        if spec.ndim != 3:
             raise ValueError(f"Expected input spec to have 3 dimensions (B, N_freq, T), but got {spec.ndim}")
        B, N_freq, T = spec.shape

        # 验证频率维度
        expected_n_freq = self.n_fft // 2 + 1
        if N_freq != expected_n_freq:
            raise ValueError(f"Input spec frequency dimension ({N_freq}) does not match n_fft ({self.n_fft}). Expected {expected_n_freq}.")

        # 逆 FFT
        # irfft 输出长度为 n_fft
        ifft_frames = jax.numpy.fft.irfft(spec, n=self.n_fft, axis=1, norm="backward") # shape: (B, n_fft, T)

        # 截断或填充 ifft 输出以匹配 win_length
        if self.win_length < self.n_fft:
            # 如果窗口比 FFT 短，通常取中间部分或开始部分。
            # librosa ISTFT (as of <0.10) effectively uses the start.
            # Let's match that behavior: take the first win_length samples.
            ifft_frames = ifft_frames[:, :self.win_length, :] # shape: (B, win_length, T)
        elif self.win_length > self.n_fft:
             # This case is invalid and handled in setup, but double check just in case.
             raise ValueError("win_length should not be greater than n_fft")
        # If win_length == n_fft, shape remains (B, n_fft, T) == (B, win_length, T)

        # 应用窗口
        # window shape: (win_length,) -> (1, win_length, 1) for broadcasting
        window = jnp.hanning(self.win_length)
        windowed_ifft = ifft_frames * window[None, :, None] # shape: (B, win_length, T)

        # 计算 OLA 输出长度（修剪前）
        output_size = (T - 1) * self.hop_length + self.win_length

        # --- 向量化重叠相加 ---
        y = self._vectorized_overlap_add(windowed_ifft, T, output_size)

        # --- 计算窗口包络 ---
        # window_sq shape: (win_length,)
        window_sq = window ** 2
        # 扩展 window_sq 以匹配 ifft_frames 的形状，用于 OLA
        # Shape: (1, win_length, T)
        window_sq_frames = jnp.broadcast_to(window_sq[None, :, None], (1, self.win_length, T))

        # 对窗口平方进行 OLA
        # 注意：窗口包络与批次无关，因此我们只计算一次（使用 window_sq_frames 的第一个元素）
        # 然后将其广播到批次维度。或者直接用 vmap 计算，结果应该相同。
        # 为了代码简洁，我们使用 vmap 并广播 window_sq_frames
        window_sq_frames_batch = jnp.broadcast_to(window_sq_frames, (B, self.win_length, T))
        window_envelope = self._vectorized_overlap_add(window_sq_frames_batch, T, output_size)

        # 归一化
        y = y / (window_envelope + self.epsilon)

        # --- 根据填充类型进行修剪 ---
        if self.padding == "center":
            # librosa 的 center padding 在 STFT 前添加了 n_fft // 2
            # 因此 ISTFT 后需要从两端移除 n_fft // 2
            pad_amount = self.n_fft // 2
            # 检查输出是否足够长以进行修剪
            if y.shape[1] < 2 * pad_amount:
                 raise ValueError(f"Output length ({y.shape[1]}) is too short for center padding trim ({pad_amount} from each end).")
            return y[:, pad_amount:-pad_amount]

        elif self.padding == "same":
            # "same" 填充（模仿某些库的行为）旨在移除窗口斜坡效应
            # 这通常对应于移除 (win_length - hop_length) // 2
            pad_amount = (self.win_length - self.hop_length) // 2
            if pad_amount < 0:
                 # 如果 hop_length > win_length，这可能为负，应为 0
                 pad_amount = 0
            # 检查输出是否足够长以进行修剪
            if y.shape[1] < 2 * pad_amount:
                 # This might happen with very short inputs, return as is or error?
                 # Let's return as is, but a warning might be good in practice.
                 print(f"Warning: Output length ({y.shape[1]}) is short relative to 'same' padding trim amount ({pad_amount}). Returning untrimmed signal.")
                 return y # Or potentially y[:, :y.shape[1] - 2*pad_amount] if we want defined length?
                         # Returning y is safer if length is less than trim amount.
            return y[:, pad_amount:-pad_amount]
        else:
             # Should not happen due to setup check, but defensive coding
             raise ValueError("Invalid padding type.") #

    def _vectorized_overlap_add(self, frames, T, output_size):
        """
        使用 jax.lax.segment_sum 实现向量化的重叠相加。

        参数:
            frames (jax.Array): 应用窗口后的 IFFT 帧，形状 (B, win_length, T)。
            T (int): 时间帧的数量。
            output_size (int): OLA 的总输出长度（修剪前）。

        返回:
            jax.Array: 重叠相加后的信号，形状 (B, output_size)。
        """
        B, N, T_check = frames.shape
        assert T == T_check
        assert N == self.win_length

        # 准备 segment_sum 输入
        # 1. 数据 (values): 将帧展平
        #    将 (B, N, T) -> (B, T, N) -> (B, T*N)
        vals = jnp.swapaxes(frames, 1, 2).reshape(B, -1)

        # 2. 段 ID (indices): 计算每个值在输出数组中的目标索引
        #    创建一个 (T, N) 的索引矩阵，其中 entry (t, n) 是 t * hop_length + n
        n_coords = jnp.arange(self.win_length)                     # (N,)
        t_starts = jnp.arange(T) * self.hop_length                # (T,)
        output_indices = t_starts[:, None] + n_coords[None, :]    # (T, N) using broadcasting
        output_indices_flat = output_indices.ravel()              # (T*N,)

        # 3. 将索引广播到批次维度
        #    segment_sum 通常在最后一个轴操作，但 vmap 更直接
        #    我们将 vmap 应用于单个批次的处理函数

        # 定义处理单个批次的函数
        def ola_single_batch(batch_vals):
            # batch_vals shape: (T*N,)
            # output_indices_flat shape: (T*N,)
            # num_segments: 总输出长度
            return jax.ops.segment_sum(batch_vals,
                                       output_indices_flat,
                                       num_segments=output_size,
                                       indices_are_sorted=False) # 索引不保证排序

        # 使用 vmap 对批次维度进行向量化
        # 对 vals 的第一个维度 (B) 进行映射
        y = jax.vmap(ola_single_batch)(vals) # Output shape: (B, output_size)
        return y
    
class ISTFTHead(nnx.Module):
    """
    ISTFT 头部模块，用于预测 STFT 复系数。

    参数:
        dim (int): 模型的隐藏维度。
        n_fft (int): 傅里叶变换的大小。
        hop_length (int): 相邻滑动窗口帧之间的距离。
        padding (str, optional): 填充类型，"center" 或 "same"。默认为 "same"。
    """

    def __init__(self, dim: int = 512, n_fft: int = 1024, hop_length: int = 256,
                 padding: str = "same", *, rngs: nnx.Rngs | None = None):
        self.dim = dim
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.padding = padding
        out_dim = self.n_fft + 2
        r = rngs or nnx.Rngs(0)
        self.out = nnx.Linear(self.dim, out_dim, rngs=r)
        self.istft = ISTFT(n_fft=self.n_fft, hop_length=self.hop_length,
                           win_length=self.n_fft, padding=self.padding, rngs=r)

    def __call__(self, x):
        # 输出维度线性投影
        x = self.out(x)
        # 转置张量维度: (B, L, H) -> (B, H, L)
        x = jnp.transpose(x, (0, 2, 1))
        # 分割幅度和相位
        mag, p = jnp.split(x, 2, axis=1)
        # 计算幅度
        mag = jnp.exp(mag)
        mag = jnp.clip(mag, a_max=1e2)  # 限制最大值
        # 计算实部和虚部
        x = jnp.cos(p)
        y = jnp.sin(p)
        # 构造复数谱
        S = mag * (x + 1j * y)
        # 调用 ISTFT 重建音频
        audio = self.istft(S)
        return audio
    
class ConvNeXtBlock(nnx.Module):
    """
    ConvNeXt 块，适配为 1D 音频信号。

    参数:
        dim (int): 输入通道数。
        intermediate_dim (int): 中间层维度。
        layer_scale_init_value (float): 层缩放的初始值，若为 0 则不应用缩放。
        adanorm_num_embeddings (int, 可选): AdaLayerNorm 的嵌入数量，若为 None 则使用普通 LayerNorm。
    """

    def __init__(self, dim: int, intermediate_dim: int, layer_scale_init_value: float,
                 adanorm_num_embeddings: int | None = None, *, rngs: nnx.Rngs | None = None):
        self.dim = dim
        self.intermediate_dim = intermediate_dim
        self.layer_scale_init_value = layer_scale_init_value
        self.adanorm_num_embeddings = adanorm_num_embeddings
        r = rngs or nnx.Rngs(0)
        # 深度卷积 (channels-last)
        self.dwconv = nnx.Conv(in_features=self.dim, out_features=self.dim, kernel_size=(7,),
                                  padding='SAME', feature_group_count=self.dim, rngs=r)
        # 归一化
        if self.adanorm_num_embeddings is not None:
            self.norm = AdaLayerNorm(self.adanorm_num_embeddings, self.dim, rngs=r)
        else:
            self.norm = nnx.LayerNorm(num_features=self.dim, epsilon=1e-6, rngs=r)
        # 逐点线性
        self.pwconv1 = nnx.Linear(self.dim, self.intermediate_dim, rngs=r)
        self.pwconv2 = nnx.Linear(self.intermediate_dim, self.dim, rngs=r)
        # 层缩放参数
        #self.gamma = nnx.Param(None)
        if self.layer_scale_init_value and self.layer_scale_init_value > 0:
            self.gamma = nnx.Param(self.layer_scale_init_value * jnp.ones((self.dim,)))

    def __call__(self, x, cond_embedding_id=None):
        residual = x
        x = self.dwconv(x)
        if self.adanorm_num_embeddings is not None:
            assert cond_embedding_id is not None
            x = self.norm(x, cond_embedding_id)
        else:
            x = self.norm(x)
        x = self.pwconv1(x)
        x = jax.nn.gelu(x)
        x = self.pwconv2(x)
        if self.layer_scale_init_value and self.layer_scale_init_value > 0:
            x = self.gamma * x
        # 残差连接
        x = residual + x
        return x


class AdaLayerNorm(nnx.Module):
    """
    自适应层归一化模块，支持基于条件嵌入的缩放和偏移。

    参数:
        num_embeddings (int): 嵌入的数量。
        embedding_dim (int): 嵌入的维度。
        eps (float): 归一化的稳定性参数，默认为 1e-6。
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, eps: float = 1e-6,
                 *, rngs: nnx.Rngs | None = None):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.eps = eps
        # 查表参数
        self.scale_table = nnx.Param(jnp.ones((self.num_embeddings, self.embedding_dim)))
        self.shift_table = nnx.Param(jnp.zeros((self.num_embeddings, self.embedding_dim)))
        self.ln = nnx.LayerNorm(num_features=self.embedding_dim, epsilon=self.eps,
                                rngs=rngs or nnx.Rngs(0))

    def __call__(self, x, cond_embedding_id):
        scale = self.scale_table[cond_embedding_id]
        shift = self.shift_table[cond_embedding_id]
        x = self.ln(x)
        return x * scale + shift
    
class VocosBackbone(nnx.Module):
    """
    Vocos 主干网络，基于 ConvNeXt 块构建，支持自适应层归一化条件。

    参数:
        input_channels (int): 输入特征通道数。
        dim (int): 模型的隐藏维度。
        intermediate_dim (int): ConvNeXtBlock 的中间维度。
        num_layers (int): ConvNeXtBlock 的层数。
        layer_scale_init_value (float, 可选): 层缩放初始值，默认为 1/num_layers。
        adanorm_num_embeddings (int, 可选): AdaLayerNorm 的嵌入数量，若为 None 则非条件模型。
    """
    def __init__(self, input_channels: int = 100, dim: int = 512, intermediate_dim: int = 1536,
                 num_layers: int = 8, layer_scale_init_value: float | None = None,
                 adanorm_num_embeddings: int | None = None, *, rngs: nnx.Rngs | None = None):
        self.input_channels = input_channels
        self.dim = dim
        self.intermediate_dim = intermediate_dim
        self.num_layers = num_layers
        self.layer_scale_init_value = layer_scale_init_value
        self.adanorm_num_embeddings = adanorm_num_embeddings
        r = rngs or nnx.Rngs(0)
        # 嵌入卷积
        self.embed = nnx.Conv(in_features=self.input_channels, out_features=self.dim,
                              kernel_size=(7,), padding='SAME', rngs=r)
        # 条件归一化或普通归一化
        if self.adanorm_num_embeddings is not None:
            self.norm = AdaLayerNorm(self.adanorm_num_embeddings, self.dim, rngs=r)
        else:
            self.norm = nnx.LayerNorm(num_features=self.dim, epsilon=1e-6, rngs=r)
        # 堆叠块
        ls = (self.layer_scale_init_value or (1 / self.num_layers))
        convnext = nnx.List([])
        for _ in range(self.num_layers):
            convnext.append(ConvNeXtBlock(self.dim, self.intermediate_dim, ls,
                          adanorm_num_embeddings=self.adanorm_num_embeddings, rngs=r))
        
        self.convnext = convnext
        # 最终归一化
        self.final_layer_norm  = nnx.LayerNorm(num_features=self.dim, epsilon=1e-6, rngs=r)

    def __call__(self, x, bandwidth_id=None):
        x = self.embed(x)
        if self.adanorm_num_embeddings is not None:
            assert bandwidth_id is not None
            x = self.norm(x, bandwidth_id)
        else:
            x = self.norm(x)
        for block in self.convnext:
            x = block(x, cond_embedding_id=bandwidth_id)
        x = self.final_layer_norm(x)
        return x
class Vocos(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs | None = None):
        r = rngs or nnx.Rngs(0)
        self.backbone = VocosBackbone(rngs=r)
        self.head = ISTFTHead(rngs=r)

    def __call__(self, x):
        x = self.backbone(x)
        audio_output = self.head(x)
        return audio_output

