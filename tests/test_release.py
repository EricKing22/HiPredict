"""Regression checks for public loading, input scaling, paths and file retention."""
from pathlib import Path
import json
import os
import subprocess

import joblib
import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from hipredict import HiPredict, get_model
from hipredict.data.normalization import load_global_normalization_stats, _resolve_global_stats_path
from hipredict.data.selection_filters import zscore_filter
from hipredict.paths import CONFIGS_DIR
from hipredict.predict.dataset import PatchInferenceDataset
from hipredict.predict.hub import download_model
from hipredict.predict.loss import get_criterion
from hipredict.predict.utils import load_cfg


@pytest.fixture
def release(tmp_path):
    cfg = load_cfg(CONFIGS_DIR / 'unet_conv_b22.yaml')
    cfg['architecture']['args'].update(base_ch=2, depth=2)
    cfg['data']['meta_cols'] = ['TDI']
    model, _ = get_model('hipredict', trained=False, cfg=cfg)
    directory = tmp_path / 'release'
    directory.mkdir()
    import yaml
    (directory / 'config.yaml').write_text(yaml.safe_dump(cfg))
    torch.save(model.state_dict(), directory / 'model.pt')
    joblib.dump(StandardScaler().fit(pd.DataFrame({'TDI': [64., 128.]})), directory / 'meta_scalar.joblib')
    (directory / 'global_stats.json').write_bytes((CONFIGS_DIR / 'global_stats.json').read_bytes())
    return directory


def test_model_alias_and_local_release(release):
    main, cfg = get_model('hipredict', model_save_dir=release)
    alias, _ = get_model('unet_conv_b22', model_save_dir=release)
    assert cfg['data']['global_stats'] == str(release / 'global_stats.json')
    assert Path(cfg['shared_meta_scalar']).is_file()
    assert all(torch.equal(value, alias.state_dict()[key]) for key, value in main.state_dict().items())


def test_missing_metadata_scaler_fails_before_inference(release):
    (release / 'meta_scalar.joblib').unlink()
    with pytest.raises(FileNotFoundError, match='meta_scalar.joblib'):
        get_model('hipredict', model_save_dir=release)


def test_hugging_face_revision_and_asset_selection(monkeypatch, release):
    calls = []
    def snapshot(**kwargs):
        calls.append(kwargs)
        return str(release)
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, 'snapshot_download', snapshot)
    assert download_model('example/hipredict', revision='pinned', local_files_only=True) == release
    assert calls[0]['revision'] == 'pinned'
    assert calls[0]['local_files_only'] is True
    assert set(calls[0]['allow_patterns']) == {'model.pt', 'meta_scalar.joblib', 'config.yaml', 'global_stats.json', 'provenance.json'}


def test_hub_defaults_to_pinned_release_and_respects_overrides(monkeypatch, release):
    from hipredict.predict import hub
    calls = []
    def snapshot(**kwargs):
        calls.append((kwargs['repo_id'], kwargs['revision']))
        return str(release)
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, 'snapshot_download', snapshot)
    monkeypatch.delenv('HIPREDICT_HF_REPO', raising=False)
    monkeypatch.delenv('HIPREDICT_HF_REVISION', raising=False)
    download_model()
    monkeypatch.setenv('HIPREDICT_HF_REPO', 'someone/fork')
    download_model()
    monkeypatch.setenv('HIPREDICT_HF_REVISION', 'v2')
    download_model()
    download_model('explicit/repo', revision='abc')
    assert calls == [(hub.DEFAULT_REPO, hub.DEFAULT_REVISION), ('someone/fork', None),
                     ('someone/fork', 'v2'), ('explicit/repo', 'abc')]


def test_release_checksum_detects_mixed_assets(release):
    (release / 'provenance.json').write_text(json.dumps({'sha256': {'model.pt': 'incorrect'}}))
    with pytest.raises(ValueError, match='checksum mismatch'):
        get_model('hipredict', model_save_dir=release)


def test_syn4_standalone_uses_physical_brightness_without_weights(monkeypatch):
    monkeypatch.delenv('HIPREDICT_HF_REPO', raising=False)
    model, cfg = get_model('SYN4')
    assert cfg['data']['meta_cols'] == []
    predictor = HiPredict(model, cfg, verbose=False)
    shape = (31, 29)
    sample = {'bg': np.full(shape, .1, dtype=np.float32),
              'ir': np.full(shape, .2, dtype=np.float32),
              'r3': np.full(shape, .4, dtype=np.float32),
              'r5': np.full(shape, .3, dtype=np.float32)}
    output = predictor.predict([sample])
    assert output.shape == (1, *shape)
    np.testing.assert_allclose(output, .3, rtol=1e-5)


def test_record_stats_do_not_override_global_normalization():
    model, cfg = get_model('SYN4')
    predictor = HiPredict(model, cfg, verbose=False)
    sample = {'x_band': np.full((2, 32, 32), .2, dtype=np.float32),
              'x_neigh': np.full((6, 128, 32), .3, dtype=np.float32),
              'obs_center': 100., 'obs_scale': 20.}
    band, _, _ = predictor._build_inputs([sample], is_explicit=True)
    center, scale = load_global_normalization_stats(_resolve_global_stats_path(cfg))
    np.testing.assert_allclose(band.numpy(), (.2 - center) / scale, rtol=1e-5)
    assert predictor.batch_stats == [(center, scale)]


def test_syn4_rejects_scene_normalization():
    model, cfg = get_model('SYN4')
    predictor = HiPredict(model, cfg, norm_mode='scene_robust', verbose=False)
    with pytest.raises(ValueError, match='requires global_robust'):
        predictor.predict([], syn4=True)


def test_config_paths_and_global_stats_do_not_depend_on_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = load_cfg(CONFIGS_DIR / 'unet_conv_b22.yaml')
    assert Path(cfg['data']['root']).is_absolute()
    assert Path(cfg['save']['model_dir']).is_absolute()
    assert _resolve_global_stats_path(cfg).is_file()


def test_historical_selection_remains_one_sided():
    frame = pd.DataFrame({'value': [0., 90., 100., 110., 200.]})
    assert zscore_filter(frame, 'value', thresh=1).tolist() == [True, True, True, True, False]


@pytest.mark.parametrize('name', ['MSELoss', 'L1Loss'])
def test_standard_losses_support_training_interface(name):
    criterion = get_criterion({'loss': {'name': name}})
    prediction = torch.ones(1, 1, 24, 24, requires_grad=True)
    loss, details = criterion(prediction, torch.zeros_like(prediction))
    loss.backward()
    assert float(loss) == 1.
    assert details and prediction.grad is not None


def test_relative_patch_manifest_can_be_loaded_outside_its_directory(tmp_path, monkeypatch):
    record_dir = tmp_path / 'dataset'
    record_dir.mkdir()
    rows = []
    for ccd in ['BG12', 'IR10', 'RED3', 'RED4', 'RED5']:
        arr = np.full((128, 32, 4) if ccd in ['RED3', 'RED5'] else (32, 32), .2, dtype=np.float32)
        np.save(record_dir / f'{ccd}.npy', arr)
        rows.append({'Set': 0, 'Observation': 'example', 'Patch': 0, 'CCD': ccd, 'Path': f'{ccd}.npy'})
    frame = pd.DataFrame(rows)
    frame.to_csv(record_dir / 'record.csv', index=False)
    monkeypatch.chdir(tmp_path)
    model, cfg = get_model('SYN4')
    samples = HiPredict(model, cfg, verbose=False).load(record_dir / 'record.csv')
    assert len(samples) == 1 and samples[0]['y'].shape == (32, 32)
    frame.attrs['record_dir'] = str(record_dir)
    assert PatchInferenceDataset(frame)[0]['obs_id'] == 'example'


def test_preprocessing_cleanup_retains_raw_and_unrelated_files(tmp_path):
    obs = 'ESP_EXAMPLE'
    keep = [f'{obs}_RED3_0.IMG', f'{obs}_RED4_processed.cub', 'notes.txt', f'{obs}_custom.cub', 'metadata.json']
    remove = [f'{obs}_RED3_0_raw.cub', f'{obs}_RED3_0_pr.cub', f'{obs}_RED3.stitched.cub']
    for name in keep + remove: (tmp_path / name).write_text('input')
    script = CONFIGS_DIR.parent / 'data/hiproc_pipe.sh'
    env = {**os.environ, 'HIPREDICT_LOG_DIR': str(tmp_path / 'logs')}
    subprocess.run(['bash', '-c', 'source "$1"; cleanup_intermediates "$2"', 'bash', str(script), obs], cwd=tmp_path, env=env, check=True)
    assert all((tmp_path / name).is_file() for name in keep)
    assert all(not (tmp_path / name).exists() for name in remove)


def test_backend_activation_failure_is_not_swallowed(tmp_path):
    script = CONFIGS_DIR.parent / 'data/activate_hiproc_backend.sh'
    env = {**os.environ, 'HIPREDICT_CONDA_SH': str(tmp_path / 'absent.sh')}
    result = subprocess.run(['bash', '-c', 'if source "$1"; then exit 0; else exit 7; fi', 'bash', str(script)], env=env, capture_output=True)
    assert result.returncode == 7
    assert b'HIPREDICT_CONDA_SH' in result.stderr


def test_download_limits_observations_and_retains_channel_files(tmp_path, monkeypatch):
    import importlib
    module = importlib.import_module('hipredict.download')
    monkeypatch.setattr(module, 'gen_edr_df', lambda: pd.DataFrame())
    seen = []
    def fetch(frame, observations, destination, workers, **kwargs):
        seen.extend(observations)
        for obs in observations:
            directory = destination / obs
            directory.mkdir()
            for ccd in kwargs['required_ccds']:
                for channel in (0, 1):
                    (directory / f'{obs}_{ccd}_{channel}.IMG').write_bytes(b'raw')
    monkeypatch.setattr(module, 'fetch_imgs_parallel', fetch)
    module.Download(['OBS_ONE', 'OBS_TWO'], tmp_path, max_obs=1, workers=1)
    assert seen == ['OBS_ONE']
    assert len(list((tmp_path / 'OBS_ONE').glob('*.IMG'))) == 10


def test_processed_cube_to_syn4_prediction_and_numeric_exports(tmp_path):
    import sys
    from hipredict.data.isis2np import isis2np
    import tifffile
    obs = 'EXAMPLE'
    directory = tmp_path / 'inputs' / obs
    directory.mkdir(parents=True)
    label = '''Object = IsisCube
  Object = Core
    StartByte = 4097
    Format = Tile
    TileSamples = 32
    TileLines = 32
    Group = Dimensions
      Samples = 32
      Lines = 32
      Bands = 1
    End_Group
    Group = Pixels
      Type = Real
      ByteOrder = Lsb
      Base = 0.0
      Multiplier = 1.0
    End_Group
  End_Object
End_Object
End
'''.encode('ascii')
    for ccd, value in [('BG12', .1), ('IR10', .2), ('RED3', .4), ('RED4', .25), ('RED5', .3)]:
        path = directory / f'{obs}_{ccd}_processed.cub'
        path.write_bytes(label + b' ' * (4096 - len(label)) + np.full((32, 32), value, dtype='<f4').tobytes())
        np.testing.assert_allclose(isis2np(str(path)), value)
    output = tmp_path / 'output'
    subprocess.run([sys.executable, '-m', 'hipredict.inference.predict', '--obs', obs,
                    '--img-dir', str(directory.parent), '--output-dir', str(output),
                    '--model', 'SYN4', '--device', 'cpu'], check=True)
    expected = np.load(output / f'{obs}_SYN4.npy')
    assert expected.dtype == np.float32
    np.testing.assert_allclose(expected, .3, rtol=1e-5)
    np.testing.assert_array_equal(tifffile.imread(output / f'{obs}_SYN4.tif'), expected)


def test_theme_plot_accepts_a_partial_research_dataset():
    pytest.importorskip('shap')
    from hipredict.post.evals.eval_plots import eval_db_plot
    from matplotlib import pyplot as plt
    categories = {'present': {'eval_category': 'theme'}, 'absent': {'eval_category': 'theme'}}
    fig, axis = eval_db_plot(categories, {'present': {'NMAE_CNN': 1., 'NMAE_BASE': 2.}}, 'hipredict')
    assert [label.get_text() for label in axis.get_xticklabels()] == ['present']
    plt.close(fig)


def test_inference_defaults_to_recorded_global_stats_for_patches_and_strips():
    from hipredict.data.normalization import normalization_kwargs_from_config
    model, cfg = get_model('SYN4')
    cfg['data'].pop('normalisation_mode')
    predictor = HiPredict(model, cfg, verbose=False)
    defaults = normalization_kwargs_from_config({})
    assert predictor.norm_mode == defaults['norm_mode'] == 'global_robust'
    for sample, explicit in [
        ({'x_band': np.full((2, 32, 32), .2), 'x_neigh': np.full((6, 128, 32), .3)}, True),
        ({key: np.full((32, 32), .2) for key in ('bg', 'ir', 'r3', 'r5')}, False),
    ]:
        sample.update(obs_center=100., obs_scale=20.)
        bands, _, _ = predictor._build_inputs([sample], is_explicit=explicit)
        np.testing.assert_allclose(bands.numpy(), (.2 - defaults['global_center']) / defaults['global_scale'], rtol=1e-5)


@pytest.mark.parametrize('dataset_type', ['HiRISEDataset', 'FilteredHiRISEDataset'])
def test_training_dataset_default_matches_global_inference(tmp_path, dataset_type):
    from hipredict.predict import dataset as module
    from hipredict.data.normalization import normalization_kwargs_from_config
    rows = []
    for ccd in ['BG12', 'IR10', 'RED3', 'RED4', 'RED5']:
        arr = np.full((128, 32, 4) if ccd in ['RED3', 'RED5'] else (32, 32), .2, dtype=np.float32)
        path = tmp_path / f'{ccd}.npy'
        np.save(path, arr)
        rows.append({'Set': 0, 'Observation': 'example', 'Patch': 0, 'CCD': ccd,
                     'Path': str(path), 'Pix_min': 0., 'Pix_max': 1., 'Date': '2020-01-01',
                     'Obs_center': 100., 'Obs_scale': 20.})
    frame = pd.DataFrame(rows)
    kwargs = {'sweep': False, 'allowed_sets': [0]} if dataset_type == 'FilteredHiRISEDataset' else {}
    sample = getattr(module, dataset_type)(frame, **kwargs)[0]
    defaults = normalization_kwargs_from_config({})
    np.testing.assert_allclose(sample['x_band'], (.2 - defaults['global_center']) / defaults['global_scale'], rtol=1e-5)
    np.testing.assert_allclose(sample['stats'], [defaults['global_center'], defaults['global_scale']])


def test_reference_generation_uses_separate_outputs(tmp_path, monkeypatch):
    from hipredict.data import compute_global_stats as stats, img_selection as selection
    runtime = tmp_path / 'files'
    runtime.mkdir()
    stats_reference = runtime / 'global_stats.json'
    list_reference = runtime / 'all_obs.txt'
    stats_reference.write_text('fixed statistics')
    list_reference.write_text('REFERENCE\n')
    monkeypatch.setattr(stats, 'files_dir', lambda: runtime)
    monkeypatch.setattr(selection, 'files_dir', lambda: runtime)
    output = stats.write_global_stats({'normalization': {'center': 42.}})
    assert output == runtime / 'generated/statistics/global_stats.json'
    assert json.loads(output.read_text())['normalization']['center'] == 42.
    calls = []
    def compute(**kwargs):
        calls.append(kwargs)
        return {'normalization': {'center': 42.}}
    monkeypatch.setattr(stats, 'compute_global_stats', compute)
    stats.main(['--img-dir', str(tmp_path / 'processed'), '--workers', '1'])
    assert calls[0]['data_record'] is None and calls[0]['workers'] == 1
    output = selection.write_observation_list(['NEW'], 'all_obs.txt')
    assert output == runtime / 'generated/selection/all_obs.txt'
    assert output.read_text() == 'NEW\n'
    with pytest.raises(ValueError, match='reference'):
        stats.write_global_stats({}, stats_reference)
    with pytest.raises(ValueError, match='reference'):
        selection.write_observation_list(['NEW'], 'all_obs.txt', runtime)
    assert stats_reference.read_text() == 'fixed statistics'
    assert list_reference.read_text() == 'REFERENCE\n'


def test_process_skip_list_is_read_only_and_fail_log_is_separate(tmp_path, monkeypatch):
    import importlib
    module = importlib.import_module('hipredict.process')
    skip = tmp_path / 'skip.txt'
    skip.write_text('SKIP\n')
    seen = []
    monkeypatch.setattr(module, 'PrepObservations', lambda **kwargs: seen.append(kwargs))
    module.main(['--obs', 'SKIP', 'KEEP', '--skip-list', str(skip), '--fail-log', str(tmp_path / 'new_failures.txt')])
    assert seen[0]['obs_list'] == ['KEEP']
    assert seen[0]['hiproc_fail_txt'] == tmp_path / 'new_failures.txt'
    with pytest.raises(SystemExit):
        module.main(['--obs', 'KEEP', '--skip-list', str(skip), '--fail-log', str(skip)])
    assert skip.read_text() == 'SKIP\n' and len(seen) == 1


def test_baseline_configuration_requires_explicit_local_configuration():
    with pytest.raises(ValueError, match='explicit cfg'):
        get_model('unet_local', trained=False)


