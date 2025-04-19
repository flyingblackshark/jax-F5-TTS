import jax.numpy as jnp
import audax
import audax.core.functional
import functools
def dynamic_range_compression_jax(x, C=1, clip_val=1e-7):
    return jnp.log(jnp.clip(x,min=clip_val) * C)

def get_mel(y, n_mels=100,n_fft=1024,win_size=1024,hop_length=256,fmin=0,fmax=None,clip_val=1e-7,sampling_rate=24000):
    window = jnp.hanning(win_size)
    spec_func = functools.partial(audax.core.functional.spectrogram, pad=0, window=window, n_fft=n_fft,
                    hop_length=hop_length, win_length=win_size, power=1.,
                    normalized=False, center=True, onesided=True)
    fb = audax.core.functional.melscale_fbanks(n_freqs=(n_fft//2)+1, n_mels=n_mels,
                        sample_rate=sampling_rate, f_min=fmin, f_max=fmax)
    mel_spec_func = functools.partial(audax.core.functional.apply_melscale, melscale_filterbank=fb)
    spec = spec_func(y)
    spec = mel_spec_func(spec)
    spec = dynamic_range_compression_jax(spec, clip_val=clip_val)
    return spec