use std::io::{self};

#[inline]
pub fn compress(uncompressed: &[u8], compressed: &mut Vec<u8>) -> io::Result<()> {
    compressed.clear();

    let mut i = 0;
    while i < uncompressed.len() {
        let byte = uncompressed[i];
        let mut run_length = 1;

        while i + run_length < uncompressed.len()
            && uncompressed[i + run_length] == byte
            && run_length < u16::MAX as usize
        {
            run_length += 1;
        }

        compressed.extend_from_slice(&(run_length as u16).to_le_bytes());
        compressed.push(byte);

        i += run_length;
    }

    Ok(())
}

#[inline]
pub fn decompress(compressed: &[u8], decompressed: &mut Vec<u8>) -> io::Result<()> {
    decompressed.clear();

    if compressed.len() % 3 != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid RLE compressed data length",
        ));
    }

    let mut i = 0;
    while i + 2 < compressed.len() {
        let run_length = u16::from_le_bytes([compressed[i], compressed[i + 1]]) as usize;
        let value = compressed[i + 2];
        for _ in 0..run_length {
            decompressed.push(value);
        }
        i += 3;
    }

    Ok(())
}
