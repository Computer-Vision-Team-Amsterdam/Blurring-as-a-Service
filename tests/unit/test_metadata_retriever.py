from blurring_as_a_service.metadata_pipeline.source.metadata_retriever import (
    Metadata,
    MetadataRetriever,
)


def expected_output_test_get_all_filenames_in_dir():
    return [
        "TMX7316010203-001252_pano_0001_000402.jpg",
        "TMX7316010203-001127_pano_0000_000139.jpg",
        "TMX7316010203-001041_pano_0002_002827.jpg",
        "TMX7316010203-001631_pano_0000_000280.jpg",
    ]


def test_get_all_filenames_in_dir():
    directory_path = "../../local_test_data/in:coco-format/sample/images/test/"
    filenames_in_dir = MetadataRetriever._get_all_filenames_in_dir(
        directory_path=directory_path
    )
    assert filenames_in_dir == expected_output_test_get_all_filenames_in_dir()


def test_filter_only_images_allowed_by_the_api():
    images_available = ["TMX2.jpg", "PANO2.jpg"]
    filenames_in_dir = MetadataRetriever._filter_only_images_allowed_by_the_api(
        images_available=images_available
    )
    assert filenames_in_dir == ["TMX2.jpg"]


def test_get_panorama_api_information():
    image_name = "TMX7316010203-001041_pano_0002_002827.jpg"
    panorama_api_information = MetadataRetriever._get_panorama_api_information(
        image_name
    )
    assert panorama_api_information == Metadata.from_dict(
        {
            "panoId": "TMX7316010203-001041_pano_0002_002827",
            "coords": [52.3659497235265, 4.867387433025],
            "geohash": "",
            "location": "",
            "cluster": 0,
            "subset": "",
        }
    )


def test_append_geohash():
    metadata_to_enhance = Metadata.from_dict(
        {
            "panoId": "TMX7316010203-001041_pano_0002_002827",
            "coords": [
                52.3659497235265,
                4.867387433025,
            ],
            "geohash": "",
            "location": "",
            "cluster": 0,
            "subset": "",
        }
    )
    metadata_with_geohash = MetadataRetriever._append_geohash(metadata_to_enhance)
    assert metadata_with_geohash == Metadata.from_dict(
        {
            "panoId": "TMX7316010203-001041_pano_0002_002827",
            "coords": [
                52.3659497235265,
                4.867387433025,
            ],
            "geohash": "u173yubmf2e1",
            "location": "",
            "cluster": 0,
            "subset": "",
        }
    )


def expected_output_test_generate_metadata():
    return [
        Metadata(
            pano_id="TMX7316010203-001252_pano_0001_000402",
            coords=(52.3162253886497, 4.97959933302371),
            geohash="u17997fh1562",
            location="",
            cluster=0,
            subset="",
        ),
        Metadata(
            pano_id="TMX7316010203-001127_pano_0000_000139",
            coords=(52.3692394303018, 4.87527034564158),
            geohash="u173yvw0fjc1",
            location="",
            cluster=0,
            subset="",
        ),
        Metadata(
            pano_id="TMX7316010203-001041_pano_0002_002827",
            coords=(52.3659497235265, 4.867387433025),
            geohash="u173yubmf2e1",
            location="",
            cluster=0,
            subset="",
        ),
        Metadata(
            pano_id="TMX7316010203-001631_pano_0000_000280",
            coords=(52.3658905697729, 4.86743740853349),
            geohash="u173yubm7t82",
            location="",
            cluster=0,
            subset="",
        ),
    ]


def test_generate_metadata():
    directory_path = "../../local_test_data/in:coco-format/sample/images/test/"
    result = MetadataRetriever(images_directory_path=directory_path).generate_metadata()
    assert result == expected_output_test_generate_metadata()
