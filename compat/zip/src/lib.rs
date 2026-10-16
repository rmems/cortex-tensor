// SPDX-License-Identifier: MIT OR Apache-2.0

//! Method facade so `candle-core` 0.11 can link `zip` 9.0.3.
//!
//! crates.io yanked `zip` 8.6.0 on 2026-10-16. `candle-core` 0.11 still
//! requires `^8.6.0` and calls `ZipFile::name() -> &str` plus
//! `ZipArchive::file_names()` items that implement `ToString`. zip 9 returns
//! `ZipResult<Cow<str>>` from both. This package is published nowhere; the
//! root `Cargo.toml` patches the crates.io package name `zip` to this path.
//! The patch version is 8.6.0 so it satisfies candle's requirement, while the
//! implementation is zip 9.0.3 (the unyanked release).

use std::borrow::Cow;
use std::io::{self, Read, Seek, Write};

pub use zip9::result;
pub use zip9::CompressionMethod;

pub mod write {
    pub use zip9::write::FileOptions;
}

/// Read handle with the zip 8 `name() -> &str` shape candle's npz reader uses.
pub struct ZipFile<'a, R: Read + ?Sized> {
    inner: zip9::read::ZipFile<'a, R>,
    name: String,
}

impl<'a, R: Read + ?Sized> ZipFile<'a, R> {
    fn wrap(inner: zip9::read::ZipFile<'a, R>) -> Self {
        let name = match inner.name() {
            Ok(name) => name.into_owned(),
            Err(_) => String::from_utf8_lossy(inner.name_raw()).into_owned(),
        };
        Self { inner, name }
    }

    /// File name as a borrowed string. Invalid UTF-8 is lossy so callers can
    /// call `strip_suffix` the way candle-core 0.11 does.
    pub fn name(&self) -> &str {
        &self.name
    }
}

impl<R: Read + ?Sized> Read for ZipFile<'_, R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.inner.read(buf)
    }

    fn read_to_end(&mut self, buf: &mut Vec<u8>) -> io::Result<usize> {
        self.inner.read_to_end(buf)
    }

    fn read_exact(&mut self, buf: &mut [u8]) -> io::Result<()> {
        self.inner.read_exact(buf)
    }
}

/// Archive reader. `file_names` yields names, not `Result`s, so
/// `.map(|f| f.to_string())` in candle's pickle reader typechecks.
pub struct ZipArchive<R> {
    inner: zip9::ZipArchive<R>,
}

impl<R: Read + Seek> ZipArchive<R> {
    /// Read a ZIP central directory.
    pub fn new(reader: R) -> zip9::result::ZipResult<Self> {
        Ok(Self {
            inner: zip9::ZipArchive::new(reader)?,
        })
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Whether the archive has no entries.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Entry at `index`.
    pub fn by_index(&mut self, index: usize) -> zip9::result::ZipResult<ZipFile<'_, R>> {
        Ok(ZipFile::wrap(self.inner.by_index(index)?))
    }

    /// Entry named `name`.
    pub fn by_name(&mut self, name: &str) -> zip9::result::ZipResult<ZipFile<'_, R>> {
        Ok(ZipFile::wrap(self.inner.by_name(name)?))
    }

    /// Entry names. A name zip 9 cannot decode becomes an empty string so the
    /// iterator item still implements `ToString`.
    pub fn file_names(&self) -> impl Iterator<Item = Cow<'_, str>> {
        self.inner.file_names().map(|name| match name {
            Ok(name) => name,
            Err(_) => Cow::Borrowed(""),
        })
    }
}

/// Archive writer. Drop finalizes the central directory, matching zip 9.
pub struct ZipWriter<W: Write + Seek> {
    inner: zip9::ZipWriter<W>,
}

impl<W: Write + Seek> ZipWriter<W> {
    /// Start an empty archive over `inner`.
    pub fn new(inner: W) -> Self {
        Self {
            inner: zip9::ZipWriter::new(inner),
        }
    }

    /// Open a stored or compressed entry. Candle passes `FileOptions<()>`.
    pub fn start_file<S: ToString>(
        &mut self,
        name: S,
        options: zip9::write::FileOptions<'_, '_, ()>,
    ) -> zip9::result::ZipResult<()> {
        self.inner.start_file(name, options)
    }
}

impl<W: Write + Seek> Write for ZipWriter<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.inner.write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::{write, CompressionMethod, ZipArchive, ZipWriter};
    use std::io::{Cursor, Read, Write};

    #[test]
    fn stored_roundtrip_matches_candle_call_shape() {
        let mut cursor = Cursor::new(Vec::new());
        {
            let mut zip = ZipWriter::new(&mut cursor);
            let options =
                write::FileOptions::default().compression_method(CompressionMethod::Stored);
            zip.start_file("weights.npy", options).unwrap();
            zip.write_all(b"npy-bytes").unwrap();
            zip.start_file("data.pkl", options).unwrap();
            zip.write_all(b"pickle").unwrap();
        }
        cursor.set_position(0);
        let mut archive = ZipArchive::new(cursor).unwrap();
        let names: Vec<String> = archive.file_names().map(|name| name.to_string()).collect();
        assert_eq!(
            names,
            vec!["weights.npy".to_string(), "data.pkl".to_string()]
        );

        let mut npy = archive.by_index(0).unwrap();
        let name = npy.name();
        assert_eq!(name.strip_suffix(".npy").unwrap_or(name), "weights");
        let mut body = String::new();
        npy.read_to_string(&mut body).unwrap();
        assert_eq!(body, "npy-bytes");
        drop(npy);

        let mut pkl = archive.by_name("data.pkl").unwrap();
        body.clear();
        pkl.read_to_string(&mut body).unwrap();
        assert_eq!(body, "pickle");
    }
}
