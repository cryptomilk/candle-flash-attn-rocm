// SPDX-License-Identifier: MIT
//
// Compiles `kernels/flash_attn_shim.cpp` and the vendored CK template
// instantiations in `kernels/generated/` with `hipcc`, then archives the
// result into `libflashattention_rocm.a`.

use anyhow::{Context, Result, bail};
use std::{
    env, fs,
    path::{Path, PathBuf},
    process::Command,
    sync::{
        Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    thread,
};

const SHIM_SOURCE: &str = "kernels/flash_attn_shim.cpp";
const SHIM_HEADER: &str = "kernels/flash_attn_shim.h";
const GENERATED_DIR: &str = "kernels/generated";
const CK_INCLUDE: &str = "composable_kernel/include";
const CK_FMHA_INCLUDE: &str = "composable_kernel/example/ck_tile/01_fmha";
const DEFAULT_ARCHES: &str = "gfx90a,gfx1100";
const LIB_NAME: &str = "flashattention_rocm";

fn main() -> Result<()> {
    println!("cargo::rerun-if-env-changed=HIPCC");
    println!("cargo::rerun-if-env-changed=ROCM_PATH");
    println!("cargo::rerun-if-env-changed=ROCM_TARGET_ARCH");
    println!("cargo::rerun-if-env-changed=CANDLE_FLASH_ATTN_ROCM_BUILD_DIR");
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed={SHIM_HEADER}");

    let sources = collect_sources()?;
    for source in &sources {
        println!("cargo::rerun-if-changed={}", source.display());
    }

    let rocm_path = find_rocm_path();
    let hipcc = find_hipcc(rocm_path.as_deref())?;
    let archs = resolve_archs(rocm_path.as_deref());

    let build_dir = resolve_build_dir()?;
    let lib_path = build_dir.join(format!("lib{LIB_NAME}.a"));
    let fingerprint_path = build_dir.join("fingerprint.txt");

    let fingerprint = compute_fingerprint(&sources, &archs)?;
    let up_to_date = lib_path.exists()
        && fs::read_to_string(&fingerprint_path)
            .is_ok_and(|existing| existing.trim() == fingerprint);

    if !up_to_date {
        let objects = compile_sources(&hipcc, &sources, &archs, &build_dir)?;
        archive(rocm_path.as_deref(), &objects, &lib_path)?;
        fs::write(&fingerprint_path, &fingerprint)
            .with_context(|| format!("writing {}", fingerprint_path.display()))?;
    }

    println!("cargo::rustc-link-search=native={}", build_dir.display());
    println!("cargo::rustc-link-lib=static={LIB_NAME}");
    if let Some(rocm_path) = &rocm_path {
        println!(
            "cargo::rustc-link-search=native={}",
            rocm_path.join("lib").display()
        );
    }
    println!("cargo::rustc-link-lib=dylib=amdhip64");
    println!("cargo::rustc-link-lib=dylib=stdc++");

    Ok(())
}

/// The C shim plus every vendored CK template instantiation, in a stable
/// (sorted) order so the fingerprint doesn't depend on directory iteration
/// order.
fn collect_sources() -> Result<Vec<PathBuf>> {
    let mut sources = vec![PathBuf::from(SHIM_SOURCE)];
    let mut generated: Vec<PathBuf> = fs::read_dir(GENERATED_DIR)
        .with_context(|| format!("reading {GENERATED_DIR}"))?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "cpp"))
        .collect();
    generated.sort();
    sources.extend(generated);
    Ok(sources)
}

fn find_rocm_path() -> Option<PathBuf> {
    if let Ok(path) = env::var("ROCM_PATH") {
        return Some(PathBuf::from(path));
    }
    let default = PathBuf::from("/opt/rocm");
    default.is_dir().then_some(default)
}

fn find_hipcc(rocm_path: Option<&Path>) -> Result<PathBuf> {
    if let Ok(path) = env::var("HIPCC") {
        return Ok(PathBuf::from(path));
    }
    if let Some(rocm_path) = rocm_path {
        let candidate = rocm_path.join("bin").join("hipcc");
        if candidate.is_file() {
            return Ok(candidate);
        }
    }
    which_on_path("hipcc").ok_or_else(|| {
        anyhow::anyhow!(
            "hipcc not found: set HIPCC to its path, or ROCM_PATH to a ROCm install, \
             or install ROCm to /opt/rocm (candle-flash-attn-rocm needs hipcc to \
             compile the CK kernels)"
        )
    })
}

fn which_on_path(name: &str) -> Option<PathBuf> {
    let paths = env::var_os("PATH")?;
    env::split_paths(&paths)
        .map(|dir| dir.join(name))
        .find(|candidate| candidate.is_file())
}

/// `ROCM_TARGET_ARCH` (comma-separated) takes priority, then `rocminfo`
/// auto-detection, then a default covering the vendored gfx9/gfx11-family
/// instantiations.
fn resolve_archs(rocm_path: Option<&Path>) -> Vec<String> {
    if let Ok(archs) = env::var("ROCM_TARGET_ARCH") {
        let archs: Vec<String> = archs
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(String::from)
            .collect();
        if !archs.is_empty() {
            return archs;
        }
    }
    if let Some(archs) = detect_archs(rocm_path)
        && !archs.is_empty()
    {
        return archs;
    }
    DEFAULT_ARCHES.split(',').map(String::from).collect()
}

fn detect_archs(rocm_path: Option<&Path>) -> Option<Vec<String>> {
    let rocminfo = rocm_path
        .map(|path| path.join("bin").join("rocminfo"))
        .filter(|path| path.is_file())
        .unwrap_or_else(|| PathBuf::from("rocminfo"));
    let output = Command::new(&rocminfo).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut archs = Vec::new();
    for line in stdout.lines() {
        let (_, value) = line.split_once("Name:")?;
        let value = value.trim();
        if value.starts_with("gfx") && !archs.contains(&value.to_string()) {
            archs.push(value.to_string());
        }
    }
    Some(archs)
}

fn resolve_build_dir() -> Result<PathBuf> {
    if let Ok(dir) = env::var("CANDLE_FLASH_ATTN_ROCM_BUILD_DIR") {
        return PathBuf::from(&dir).canonicalize().with_context(|| {
            format!(
                "CANDLE_FLASH_ATTN_ROCM_BUILD_DIR is set to {dir}, but that directory doesn't exist"
            )
        });
    }
    let out_dir = env::var("OUT_DIR").context("OUT_DIR not set")?;
    Ok(PathBuf::from(out_dir))
}

/// FNV-1a over the shim header, every source file's path and contents, and
/// the resolved arch list — changing any of these invalidates the cache.
fn compute_fingerprint(sources: &[PathBuf], archs: &[String]) -> Result<String> {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    fnv_update(
        &mut hash,
        &fs::read(SHIM_HEADER).context("reading flash_attn_shim.h")?,
    );
    for source in sources {
        fnv_update(&mut hash, source.to_string_lossy().as_bytes());
        fnv_update(
            &mut hash,
            &fs::read(source).with_context(|| format!("reading {}", source.display()))?,
        );
    }
    for arch in archs {
        fnv_update(&mut hash, arch.as_bytes());
    }
    Ok(format!("{hash:016x}"))
}

fn fnv_update(hash: &mut u64, bytes: &[u8]) {
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
    for &byte in bytes {
        *hash ^= u64::from(byte);
        *hash = hash.wrapping_mul(FNV_PRIME);
    }
}

/// Compiles every source to a `.o` in `<build_dir>/obj`, spawning `hipcc`
/// across roughly half the available CPUs (full CK template instantiation
/// compiles are slow; hipcc itself only compiles one file at a time).
fn compile_sources(
    hipcc: &Path,
    sources: &[PathBuf],
    archs: &[String],
    build_dir: &Path,
) -> Result<Vec<PathBuf>> {
    let obj_dir = build_dir.join("obj");
    fs::create_dir_all(&obj_dir).with_context(|| format!("creating {}", obj_dir.display()))?;

    let jobs: Vec<(PathBuf, PathBuf)> = sources
        .iter()
        .map(|source| {
            let stem = source
                .file_stem()
                .context("source file has no stem")?
                .to_string_lossy()
                .into_owned();
            Ok((source.clone(), obj_dir.join(format!("{stem}.o"))))
        })
        .collect::<Result<_>>()?;

    let worker_count = thread::available_parallelism()
        .map_or(1, std::num::NonZero::get)
        .div_ceil(2)
        .max(1);
    let next_job = AtomicUsize::new(0);
    let errors: Mutex<Vec<String>> = Mutex::new(Vec::new());

    thread::scope(|scope| {
        for _ in 0..worker_count {
            scope.spawn(|| {
                loop {
                    let index = next_job.fetch_add(1, Ordering::SeqCst);
                    let Some((source, object)) = jobs.get(index) else {
                        break;
                    };
                    if let Err(err) = compile_one(hipcc, source, object, archs) {
                        errors
                            .lock()
                            .unwrap()
                            .push(format!("{}: {err}", source.display()));
                    }
                }
            });
        }
    });

    let errors = errors.into_inner().unwrap();
    if !errors.is_empty() {
        bail!(
            "hipcc failed for {} source(s):\n{}",
            errors.len(),
            errors.join("\n")
        );
    }

    Ok(jobs.into_iter().map(|(_, object)| object).collect())
}

fn compile_one(hipcc: &Path, source: &Path, object: &Path, archs: &[String]) -> Result<()> {
    let mut cmd = Command::new(hipcc);
    cmd.arg("-std=c++20")
        .arg("-O3")
        .arg("-fPIC")
        .arg("-fbracket-depth=1024")
        .arg("-Wno-undefined-func-template")
        .arg("-Wno-float-equal")
        .arg("-DCK_TILE_FMHA_FWD_FAST_EXP2=1")
        .arg("-fgpu-flush-denormals-to-zero")
        .arg("-I")
        .arg(CK_INCLUDE)
        .arg("-I")
        .arg(CK_FMHA_INCLUDE)
        .arg("-I")
        .arg("kernels")
        .arg("-c")
        .arg(source)
        .arg("-o")
        .arg(object);
    for arch in archs {
        cmd.arg(format!("--offload-arch={arch}"));
    }
    let status = cmd
        .status()
        .with_context(|| format!("spawning {}", hipcc.display()))?;
    if !status.success() {
        bail!("hipcc exited with {status}");
    }
    Ok(())
}

fn archive(rocm_path: Option<&Path>, objects: &[PathBuf], lib_path: &Path) -> Result<()> {
    if lib_path.exists() {
        fs::remove_file(lib_path)
            .with_context(|| format!("removing stale {}", lib_path.display()))?;
    }
    let ar = find_ar(rocm_path);
    let status = Command::new(&ar)
        .arg("crs")
        .arg(lib_path)
        .args(objects)
        .status()
        .with_context(|| format!("spawning {}", ar.display()))?;
    if !status.success() {
        bail!("{} exited with {status}", ar.display());
    }
    Ok(())
}

fn find_ar(rocm_path: Option<&Path>) -> PathBuf {
    if let Some(rocm_path) = rocm_path {
        let candidate = rocm_path.join("llvm").join("bin").join("llvm-ar");
        if candidate.is_file() {
            return candidate;
        }
    }
    which_on_path("llvm-ar").unwrap_or_else(|| PathBuf::from("ar"))
}
