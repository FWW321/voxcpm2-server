use anyhow::Result;
use candle_core::{D, DType, Device, Tensor};

pub fn linspace(start: f32, end: f32, steps: usize, device: &Device) -> Result<Tensor> {
    assert!(steps > 0, "steps must be > 0");
    if steps == 1 {
        return Ok(Tensor::from_slice(&[start], 1, device)?);
    }
    let step_size = (end - start) / (steps - 1) as f32;
    let data: Vec<f32> = (0..steps).map(|i| start + i as f32 * step_size).collect();
    Ok(Tensor::from_slice(&data, steps, device)?)
}

pub fn prepare_causal_attention_mask(
    b_size: usize,
    tgt_len: usize,
    seqlen_offset: usize,
    device: &Device,
) -> Result<Tensor> {
    let arange = Tensor::arange(0u32, tgt_len as u32, device)?;
    let arange = arange.unsqueeze(1)?.broadcast_as((tgt_len, tgt_len))?;
    let upper_triangle = arange.t()?.gt(&arange)?;
    let mask = upper_triangle.where_cond(
        &Tensor::new(f32::NEG_INFINITY, device)?.broadcast_as(arange.shape())?,
        &Tensor::new(0f32, device)?.broadcast_as(arange.shape())?,
    )?;
    let mask = if seqlen_offset > 0 {
        let mask0 = Tensor::zeros((tgt_len, seqlen_offset), DType::F32, device)?;
        Tensor::cat(&[&mask0, &mask], D::Minus1)?
    } else {
        mask
    };
    Ok(mask
        .expand((b_size, 1, tgt_len, tgt_len + seqlen_offset))?
        .to_dtype(DType::F32)?)
}

pub fn repeat_kv(xs: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        Ok(xs)
    } else {
        let (b_sz, n_kv_head, seq_len, head_dim) = xs.dims4()?;
        Ok(Tensor::cat(&vec![&xs; n_rep], 2)?.reshape((
            b_sz,
            n_kv_head * n_rep,
            seq_len,
            head_dim,
        ))?)
    }
}
