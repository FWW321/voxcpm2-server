pub mod rope;
pub mod tensor;

use anyhow::{Result, anyhow};

pub fn bucketize(input: usize, boundaries: &[usize]) -> Result<usize> {
    if boundaries.is_empty() {
        return Err(anyhow!("bucketize param boundaries can not be empty"));
    }
    match boundaries.binary_search(&input) {
        Ok(i) => Ok(i),
        Err(i) => Ok(i),
    }
}

pub fn get_device(device: Option<&candle_core::Device>) -> candle_core::Device {
    match device {
        Some(d) => d.clone(),
        None => {
            #[cfg(feature = "cuda")]
            {
                candle_core::Device::new_cuda(0).unwrap_or(candle_core::Device::Cpu)
            }
            #[cfg(not(feature = "cuda"))]
            {
                candle_core::Device::Cpu
            }
        }
    }
}

pub fn get_dtype(dtype: Option<candle_core::DType>, cfg_dtype: &str) -> candle_core::DType {
    match dtype {
        Some(d) => d,
        None => match cfg_dtype {
            "float32" | "float" => candle_core::DType::F32,
            "float64" | "double" => candle_core::DType::F64,
            "float16" => candle_core::DType::F16,
            "bfloat16" => candle_core::DType::BF16,
            _ => candle_core::DType::F32,
        },
    }
}

pub fn find_type_files(path: &str, extension_type: &str) -> anyhow::Result<Vec<String>> {
    let mut files = Vec::new();
    for entry in std::fs::read_dir(path)? {
        let entry = entry?;
        let file_path = entry.path();
        if file_path.is_file()
            && let Some(extension) = file_path.extension()
            && extension == extension_type
        {
            files.push(file_path.to_string_lossy().to_string());
        }
    }
    Ok(files)
}
