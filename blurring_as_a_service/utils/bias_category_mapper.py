from dataclasses import dataclass
from typing import List


@dataclass
class SensitiveCategories:
    values: List[str] = None

    def __post_init__(self):
        if self.values is None:
            self.values = [
                "grouped_category",
                "sex",
                "age",
                "skin_color",
                "license_plate_origin",
                "license_plate_color",
            ]


class BiasCategoryMapper:
    def __init__(self, categories: List[dict]):
        """
        Group categories into person or license plates based on the original category name.

        Parameters
        ----------
        categories
            Original category name, example:
                [{"id": 1, "name": "man"}, {"id": 2, "name": "man/child"}, {"id": 3, "name": "man/child/light"}]

        Returns
        -------
            Grouped category, example: [
                {"id": 1, "name": "man", 'grouped_category': 0},
                {"id": 2, "name": "man/child", 'grouped_category': 0},
                {"id": 3, "name": "man/child/light", 'grouped_category': 0}
            ]
        """
        for category in categories:
            if category["name"].startswith("license_plate"):
                category["grouped_category"] = 1
            else:
                category["grouped_category"] = 0
        self._grouped_categories = categories

    def get_all_grouped_categories(self):
        return self._grouped_categories

    def get_grouped_category(self, category_id) -> int:
        """
        Returns the correct grouped category for a specific annotation.

        Parameters
        ----------
        category_id:
            Original category.

        Returns
        -------
        Grouped category.

        """
        return list(
            filter(lambda cat: cat["id"] == category_id, self._grouped_categories)
        )[0]["grouped_category"]
