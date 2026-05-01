"""NICE-class to industry-name catalogue.

The first 34 classes are goods, the next 11 are services. Names are
abbreviated industry headings drawn from the WIPO NICE Classification.
"""

INDUSTRY_NAME = {
    "001": "Chemicals & Industrial",
    "002": "Paints & Coatings",
    "003": "Cosmetics & Cleaning",
    "004": "Lubricants & Fuels",
    "005": "Pharmaceuticals",
    "006": "Common Metals",
    "007": "Machines & Power Tools",
    "008": "Hand Tools",
    "009": "Software & Electronics",
    "010": "Medical Devices",
    "011": "Lighting, HVAC & Appliances",
    "012": "Vehicles",
    "013": "Firearms & Pyrotechnics",
    "014": "Jewelry & Watches",
    "015": "Musical Instruments",
    "016": "Paper, Print & Office",
    "017": "Rubber & Insulators",
    "018": "Leather & Luggage",
    "019": "Building Materials",
    "020": "Furniture",
    "021": "Household Utensils",
    "022": "Ropes, Tents & Textiles (raw)",
    "023": "Yarns & Threads",
    "024": "Textiles & Bedding",
    "025": "Clothing, Footwear & Headwear",
    "026": "Trim, Lace & Notions",
    "027": "Carpets & Wall Hangings",
    "028": "Toys, Games & Sporting Goods",
    "029": "Meat, Dairy & Prepared Foods",
    "030": "Coffee, Tea, Bakery & Spices",
    "031": "Agriculture & Live Animals",
    "032": "Beer & Soft Drinks",
    "033": "Wine & Spirits",
    "034": "Tobacco & Smokers' Articles",
    "035": "Advertising & Retail",
    "036": "Insurance & Financial",
    "037": "Construction & Repair",
    "038": "Telecommunications",
    "039": "Transport & Logistics",
    "040": "Materials Treatment",
    "041": "Education & Entertainment",
    "042": "Scientific & Tech Services",
    "043": "Restaurants & Hotels",
    "044": "Medical & Personal Care",
    "045": "Legal & Personal Services",
}

INDUSTRY_SHORT = {
    "009": "software",
    "032": "beverages",
}


def name(cls: str) -> str:
    return INDUSTRY_NAME.get(cls, f"NICE Class {cls}")


def all_classes() -> list[str]:
    return [f"{i:03d}" for i in range(1, 46)]
