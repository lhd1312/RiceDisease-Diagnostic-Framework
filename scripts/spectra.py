"""Pair sensor workbooks and apply observation-wise Savitzky–Golay and SNV."""
import numpy as np
from openpyxl import load_workbook
from scipy.signal import savgol_filter


def read_sensor(file):
    wb = load_workbook(file, read_only=True, data_only=True)
    with_data = {}
    names = wb.sheetnames[:-1]
    bands = []
    for name in names:
        values = np.asarray(list(wb[name].values), dtype=np.float64)
        if len(bands) == 0:
            bands = values[0]
        np.testing.assert_array_equal(bands, values[0])
        with_data[name] = values[1:]
    metadata = list(wb["盆号对应表"].values)[1:]
    wb.close()
    return bands, with_data, metadata, names


def load_data(data_dir, expected_rows=1005, expected_pots=181, retain_transition=False):
    v_bands, v, v_meta, names = read_sensor(data_dir / "VNIR.xlsx")
    s_bands, s, s_meta, s_names = read_sensor(data_dir / "SWIR.xlsx")
    assert v_meta == s_meta and names == s_names
    assert len(v_meta) == expected_rows and len({row[3] for row in v_meta}) == expected_pots
    assert len({row[2] for row in v_meta}) == len(v_meta)
    assert len({(row[3], row[-2]) for row in v_meta}) == len(v_meta)
    assert all(row[2].startswith("OBS_") for row in v_meta)
    assert all(row[-1] == names.index(row[0]) for row in v_meta)
    assert {(sheet, line) for sheet, line, *_ in v_meta} == {
        (sheet, line) for sheet in names for line in range(2, len(v[sheet]) + 2)}
    assert all(len(v[sheet]) == len(s[sheet]) for sheet in names)
    x = np.asarray([np.concatenate((v[sheet][line-2], s[sheet][line-2]))
                    for sheet, line, *_ in v_meta])
    wl = np.concatenate((v_bands, s_bands))
    assert x.shape == (expected_rows, 798) and np.isfinite(x).all()
    keep = ((wl >= 420) & (wl <= 900)) | ((wl >= 1000) & (wl <= 1700))
    if retain_transition:
        keep |= (wl > 900) & (wl < 1000)
    filtered = savgol_filter(x[:, keep], window_length=11, polyorder=3, axis=1)
    x = (filtered - filtered.mean(axis=1, keepdims=True)) / (
        filtered.std(axis=1, keepdims=True) + 1e-8)
    assert np.isfinite(x).all()
    return x, np.asarray([row[-1] for row in v_meta]), v_meta, names
