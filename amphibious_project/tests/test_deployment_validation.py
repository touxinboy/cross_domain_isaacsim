"""Deployment defects: corrupt/missing files, Linux case, malformed queues, caches."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import shutil
import types
import unittest
from unittest.mock import patch

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/"scripts"))
from navigation_io import read_json,write_json
from omninav_worker import process_request
import omninav_worker
from model_environment import configure_model_environment


def load(relative,name):
    spec=importlib.util.spec_from_file_location(name,PROJECT/relative)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


preflight=load("deploy/verify_deployment.py","l17_preflight_tests")
installer=load("deploy/omninav/setup_environment.py","l17_install_tests")


class DeploymentChecks(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix="l17_deploy_checks_");self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)

    def test_manifest_catches_modified_packaged_file(self):
        path=self.root/"scripts/a.py";path.parent.mkdir();path.write_text("original")
        write_json(self.root/"PACKAGED_FILES.json",{"files":[{"path":"scripts/a.py","bytes":8,"sha256":hashlib.sha256(b"original").hexdigest()}]})
        self.assertEqual(preflight.verify_manifest(self.root,True)["status"],"passed")
        path.write_text("modified")
        with self.assertRaisesRegex(ValueError,"absent or modified"):preflight.verify_manifest(self.root,True)

    def test_model_copy_corruption_is_caught_before_importing_torch(self):
        path=self.root/"weight.bin";path.write_bytes(b"expected")
        lock={"revision":"fixture","files":[{"path":path.name,"bytes":8,"sha256":hashlib.sha256(b"expected").hexdigest()}]}
        self.assertEqual(preflight.verify_weights(self.root,lock)["status"],"passed")
        path.write_bytes(b"modified")
        with self.assertRaisesRegex(ValueError,"SHA-256 mismatch"):preflight.verify_weights(self.root,lock)

    def test_linux_letter_case_is_checked_on_windows_too(self):
        write_json(self.root/"configs/task.json",{"scene_file":"scenes/test.usda"})
        (self.root/"scenes").mkdir();(self.root/"assets").mkdir()
        (self.root/"assets/rock.usda").write_text("#usda 1.0\n")
        (self.root/"scenes/test.usda").write_text("#usda 1.0\n# @../assets/ROCK.usda@\n")
        with self.assertRaises(ValueError):preflight.verify_scene(self.root)

    def test_malformed_request_gets_a_terminal_result_instead_of_infinite_retries(self):
        path=self.root/"queue/bad.request.json";path.parent.mkdir();path.write_text("broken JSON")
        class MustNotCall:
            def predict(self,request):raise AssertionError("Invalid request must not call the model")
        reply=process_request(self.root,path,MustNotCall())
        self.assertFalse(reply["ok"]);self.assertFalse(reply["model_inference_attempted"])
        self.assertTrue(path.with_name("bad.result.json").exists())

    def test_install_plan_never_changes_system_python_or_shared_isaac(self):
        paths,commands=installer.install_plan(self.root,"python3.10")
        self.assertTrue(all(path.is_relative_to(self.root) for path in paths.values()))
        self.assertEqual(commands[0][:3],["python3.10","-m","venv"])
        self.assertTrue(all(command[0]==str(paths["environment"]/"bin/python") for command in commands[1:]))
        self.assertFalse(any("sudo" in command for command in commands))

    def test_cache_configuration_overrides_external_defaults_inside_own_process(self):
        previous=tempfile.tempdir
        try:
            with patch.dict(os.environ,{"HF_HOME":"EXTERNAL_CACHE","TRANSFORMERS_CACHE":"EXTERNAL_CACHE"}):
                folders=configure_model_environment(self.root)
                self.assertTrue(all(Path(path).is_relative_to(self.root) for path in folders.values()))
                self.assertNotIn("TRANSFORMERS_CACHE",os.environ)
                self.assertEqual(os.environ["HF_HUB_OFFLINE"],"1")
        finally:tempfile.tempdir=previous

    def test_interpreter_without_ensurepip_is_skipped(self):
        failed=types.SimpleNamespace(returncode=1,stdout="")
        working=types.SimpleNamespace(returncode=0,stdout="[3, 10]")
        with patch.object(installer.shutil,"which",side_effect=lambda value:value),patch.object(installer.subprocess,"run",side_effect=[failed,working]) as probe:
            self.assertEqual(installer.choose_python(),"python3.10")
        self.assertIn("ensurepip",probe.call_args_list[0].args[0][-1])

    def test_torch_mirror_keeps_exact_versions_and_own_environment(self):
        paths,commands=installer.install_plan(self.root,"python3.10","https://pypi.tuna.tsinghua.edu.cn/simple")
        command=commands[2]
        self.assertEqual(command[0],str(paths["environment"]/"bin/python"))
        self.assertIn("torch==2.6.0",command)
        self.assertIn("torchvision==0.21.0",command)
        self.assertEqual(command[-1],"https://pypi.tuna.tsinghua.edu.cn/simple")
        with self.assertRaises(ValueError):installer.install_plan(self.root,"python3.10","http://example.com/simple")

    def worker_tree(self):
        names=["scripts/omninav_pointgoal.py","scripts/navigation_contracts.py","configs/navigation_camera.json",
               "third_party/OmniNav/infer_ovon_slowfast/qwen_utils.py","deploy/omninav/smoke_model.py"]
        for name in names:
            target=self.root/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(PROJECT/name,target)
        target=self.root/"checkpoints/fixture/config.json";target.parent.mkdir(parents=True)
        write_json(target,{"fixture_only":True})
        # No trained weights are loaded by this worker-lifecycle fixture.
        from test_navigation_preparation import FixturePredictor
        predictor=FixturePredictor()
        predictor.model=types.SimpleNamespace(device="cpu")
        predictor.torch=types.SimpleNamespace(cuda=types.SimpleNamespace(reset_peak_memory_stats=lambda device:None,max_memory_allocated=lambda device:0))
        return predictor

    def worker_args(self,*extra):
        return ["omninav_worker.py","--project",str(self.root),"--checkpoint","checkpoints/fixture",
                "--queue","queue","--idle-exit-s","0.01",*extra]

    def test_worker_announces_ready_only_after_startup_forward(self):
        predictor=self.worker_tree()
        with patch.object(sys,"argv",self.worker_args("--startup-smoke")),patch.object(omninav_worker,"OmniNavPointGoal",return_value=predictor):
            omninav_worker.main()
        self.assertTrue(read_json(self.root/"queue/ready.json")["startup_smoke_passed"])
        self.assertTrue(read_json(self.root/"queue/startup_smoke.json")["test_fixture_only"])
        self.assertFalse((self.root/"queue/worker.lock").exists())

    def test_failed_startup_forward_never_leaves_a_ready_marker(self):
        predictor=self.worker_tree()
        predictor.predict=lambda request:(_ for _ in ()).throw(RuntimeError("startup forward failed"))
        with patch.object(sys,"argv",self.worker_args("--startup-smoke")),patch.object(omninav_worker,"OmniNavPointGoal",return_value=predictor):
            with self.assertRaisesRegex(RuntimeError,"startup forward failed"):omninav_worker.main()
        self.assertFalse((self.root/"queue/ready.json").exists())
        self.assertFalse((self.root/"queue/worker.lock").exists())
        self.assertTrue((self.root/"queue/startup_or_runtime_error.json").exists())

    def test_once_mode_reports_failure_for_a_bad_request(self):
        predictor=self.worker_tree()
        path=self.root/"queue/bad.request.json";path.parent.mkdir();path.write_text("broken JSON")
        with patch.object(sys,"argv",self.worker_args("--once")),patch.object(omninav_worker,"OmniNavPointGoal",return_value=predictor):
            result=omninav_worker.main()
        self.assertEqual(result,3)


if __name__=="__main__":unittest.main(verbosity=2)
