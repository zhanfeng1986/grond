from __future__ import absolute_import, print_function
import os
import shutil

from grond import config
from grond import Environment

from . import common
from .common import grond, chdir

_multiprocess_can_split = True


def test_cmt2():
    playground_dir = common.get_playground_dir()
    common.get_test_data('gf_stores/crust2_ib/')
    common.get_test_data('gf_stores/crust2_ib_static/')
    gf_stores_path = common.test_data_path('gf_stores')

    with chdir(playground_dir):
        scenario_dir = 'scenario_multisource'
        if os.path.exists(scenario_dir):
            shutil.rmtree(scenario_dir)

        grond('scenario', '--targets=waveforms', '--nevents=2',
              '--nstations=5', '--gf-store-superdirs=%s' % gf_stores_path,
              scenario_dir)

        with chdir(scenario_dir):
            config_path = 'config/scenario.gronf'
            quick_config_path = 'config/scenario_quick.gronf'
            event_names = grond('events', config_path).strip().split('\n')

            env = Environment([config_path] + event_names)
            conf = env.get_config()

            mod_conf = conf.clone()
            mod_conf.set_elements(
                'analyser_configs[:].niterations', 100)
            mod_conf.set_elements(
                'optimiser_config.sampler_phases[:].niterations', 100)
            mod_conf.set_elements(
                'optimiser_config.nbootstrap', 5)
            mod_conf.set_basepath(conf.get_basepath())
            config.write_config(mod_conf, quick_config_path)
            grond('diff', config_path, quick_config_path)
            grond('check', quick_config_path, *event_names)

            grond('go', quick_config_path, *event_names)
            rundir_paths = common.get_rundir_paths(config_path, event_names)
            grond('report', *rundir_paths)

            rundir_paths = common.get_rundir_paths(config_path, event_names)
            grond('report', '--parallel=2', *rundir_paths)
