import base64
import json

from blurring_as_a_service.endpoint.components.scoring import init, run
from blurring_as_a_service.settings.settings import BlurringAsAServiceSettings

BlurringAsAServiceSettings.set_from_yaml("config.yml")
settings = BlurringAsAServiceSettings.get_settings()


def main():
    with open("blurring_as_a_service/endpoint/test_image.jpg", "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    init()
    response = run(json.dumps({"data": image_b64}))

    result = json.loads(response)
    annotated_image_b64 = result.get("annotated_image")
    if annotated_image_b64:
        with open("annotated_image.jpg", "wb") as f:
            f.write(base64.b64decode(annotated_image_b64))


if __name__ == "__main__":
    main()
