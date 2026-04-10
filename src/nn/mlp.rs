use candle_core::Tensor;
use candle_nn::{Activation, Linear, Module, VarBuilder, linear_b};

use anyhow::Result;

pub struct MlpConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub activation: Activation,
    pub bias: bool,
    pub gate_proj_name: &'static str,
    pub up_proj_name: &'static str,
    pub down_proj_name: &'static str,
}

#[derive(Debug, Clone)]
pub struct GateUpDownMLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act_fn: Activation,
}

impl GateUpDownMLP {
    pub fn new(vb: VarBuilder, config: &MlpConfig) -> Result<Self> {
        let gate_proj = linear_b(
            config.hidden_size,
            config.intermediate_size,
            config.bias,
            vb.pp(config.gate_proj_name),
        )?;
        let up_proj = linear_b(
            config.hidden_size,
            config.intermediate_size,
            config.bias,
            vb.pp(config.up_proj_name),
        )?;
        let down_proj = linear_b(
            config.intermediate_size,
            config.hidden_size,
            config.bias,
            vb.pp(config.down_proj_name),
        )?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn: config.activation,
        })
    }
}

impl Module for GateUpDownMLP {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let lhs = xs.apply(&self.gate_proj)?.apply(&self.act_fn)?;
        let rhs = xs.apply(&self.up_proj)?;
        (lhs * rhs)?.apply(&self.down_proj)
    }
}
