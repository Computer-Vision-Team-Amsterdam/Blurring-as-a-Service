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
    with open("test_image.jpg", "rb") as f:
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

    payload = str.encode(json.dumps({"data": image_b64}))
    if scoring_url.lower().startswith("http"):
        req = urllib.request.Request(scoring_url, payload, headers)
    else:
        raise ValueError(f"Invalid scoring URL: {scoring_url}")
    try:
        response = urllib.request.urlopen(req).read()  # nosec
        response_dict = json.loads(json.loads(response))
        annotated_image_b64 = response_dict["annotated_image"]
        if annotated_image_b64:
            with open("annotated_image.jpg", "wb") as f:
                f.write(base64.b64decode(annotated_image_b64))
    except urllib.error.HTTPError as error:
        print("The request failed with status code: " + str(error.code))
        print(error.info())
        print(error.read().decode("utf8", "ignore"))


if __name__ == "__main__":
    main()
