
from amla_toolkit import AMLInterface
from azureml.core import Datastore, Environment, Workspace
from azureml.core.runconfig import RunConfiguration
from azureml.data import OutputFileDatasetConfig
from azureml.pipeline.steps import PythonScriptStep


def main():
    aml_interface = AMLInterface("aml-config.yml", "analyse_services", dtap="ont")
    ws = Workspace.from_config()
    env = Environment.get(workspace=ws, name="blurring-env")
    pipeline_config = RunConfiguration()
    pipeline_config.environment = env

    validate_model = PythonScriptStep(
        name="validate-model-step",
        script_name="run_validation.py",
        outputs=[labels],
        compute_target="yolo-cluster",
        source_directory=".",
        allow_reuse=True,
        runconfig=pipeline_config,
    )

    run = aml_interface.create_pipeline(
        [validate_model], "pipeline-validation"
    )
    run.wait_for_completion(show_output=True)


if __name__ == "__main__":
    main()
