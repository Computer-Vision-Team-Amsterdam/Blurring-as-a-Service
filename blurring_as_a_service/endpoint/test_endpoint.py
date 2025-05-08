import base64
import json
import os
import ssl
import urllib.request

from aml_interface.aml_interface import AMLInterface


def allowSelfSignedHttps(allowed):
    # bypass the server certificate verification on client side
    if (
        allowed
        and not os.environ.get("PYTHONHTTPSVERIFY", "")
        and getattr(ssl, "_create_unverified_context", None)
    ):
        ssl._create_default_https_context = ssl._create_unverified_context


allowSelfSignedHttps(
    True
)  # this line is needed if you use self-signed certificate in your scoring service.


def main():
    with open("blurring_as_a_service/endpoint/test_image.jpg", "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    aml_interface = AMLInterface()
    ml_client = aml_interface.ml_client

    scoring_url = ml_client.online_endpoints.get("endpt-cvt-baas-2").scoring_uri
    access_token = ml_client.online_endpoints.get_keys("endpt-cvt-baas-2").access_token
    expiry_time_utc = ml_client.online_endpoints.get_keys(
        "endpt-cvt-baas-2"
    ).expiry_time_utc
    refresh_after_time_utc = ml_client.online_endpoints.get_keys(
        "endpt-cvt-baas-2"
    ).refresh_after_time_utc
    print(
        f"scoring_url: {scoring_url}, access_token: {len(access_token)}, expiry_time_utc: {expiry_time_utc}, refresh_after_time_utc: {refresh_after_time_utc}"
    )

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/json",
    }

    payload = str.encode(
        json.dumps(
            {
                "data": image_b64,
                "user_id": "s.davrieux@amsterdam.nl",
            }
        )
    )
    req = urllib.request.Request(scoring_url, payload, headers)

    url_to_check = req.get_full_url()
    if not url_to_check.startswith(("http://", "https://")):
        print(f"Error: Invalid URL scheme in request: {url_to_check}")
        raise ValueError(
            f"Disallowed URL scheme: {url_to_check}. Only http/https are allowed."
        )

    try:
        with urllib.request.urlopen(req) as http_response:  # nosec B310
            status_code = http_response.getcode()
            response_body_str = http_response.read().decode("utf-8")
            response_dict = json.loads(json.loads(response_body_str)[0])
            if status_code != 200:
                if "error" in response_dict:
                    raise ValueError(
                        f"Request failed with status code: {status_code}, error in response: {response_dict['error']}"
                    )
                raise ValueError(f"Request failed with status code: {status_code}")
            elif "annotated_image" in response_dict:
                annotated_image_b64 = response_dict["annotated_image"]
                if annotated_image_b64:
                    with open("annotated_image.jpg", "wb") as f:
                        f.write(base64.b64decode(annotated_image_b64))
            if "metadata" in response_dict:
                print("Metadata: ", response_dict["metadata"])
    except Exception as error:
        print("The process of the response failed with error code: " + str(error.code))
        print(error.info())
        print(error.read().decode("utf8", "ignore"))


if __name__ == "__main__":
    main()
