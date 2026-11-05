"""General recycling guidance for each class. Local rules vary by city."""

CLASS_INFO = {
    "battery": {
        "icon": "🔋",
        "display_name": "Battery",
        "category": "Hazardous / e-waste",
        "bin": "Battery or e-waste collection point (not the household bin)",
        "tip": "Never throw batteries in normal waste. They can leak chemicals or catch fire. "
        "Drop them at an authorised collection point.",
    },
    "biological": {
        "icon": "🍎",
        "display_name": "Biological / organic waste",
        "category": "Compostable",
        "bin": "Wet / organic waste bin or compost",
        "tip": "Food scraps and garden waste can be composted. Keep it free of plastic.",
    },
    "brown-glass": {
        "icon": "🍾",
        "display_name": "Brown glass",
        "category": "Recyclable (glass)",
        "bin": "Dry recyclable waste / glass collection",
        "tip": "Empty and rinse the container. Wrap broken glass safely before disposal.",
    },
    "cardboard": {
        "icon": "📦",
        "display_name": "Cardboard",
        "category": "Recyclable (paper fibre)",
        "bin": "Dry recyclable waste",
        "tip": "Flatten boxes and keep them dry. Heavily soiled or greasy cardboard is harder to recycle.",
    },
    "clothes": {
        "icon": "👕",
        "display_name": "Clothes / textiles",
        "category": "Reuse / donate",
        "bin": "Donation centre or textile collection",
        "tip": "Wearable clothes can be donated or reused. Worn-out textiles can go to textile recycling.",
    },
    "green-glass": {
        "icon": "🍾",
        "display_name": "Green glass",
        "category": "Recyclable (glass)",
        "bin": "Dry recyclable waste / glass collection",
        "tip": "Empty and rinse the container. Wrap broken glass safely before disposal.",
    },
    "e-waste": {
        "icon": "📱",
        "display_name": "E-waste (electronics)",
        "category": "Hazardous / e-waste",
        "bin": "Authorised e-waste collection point or manufacturer/retailer take-back",
        "tip": "Do not put electronics in the household bin. Delete personal data from phones "
        "and storage first. Keep damaged or swollen batteries separate and hand them to "
        "an authorised collector.",
    },
    "metal": {
        "icon": "🥫",
        "display_name": "Metal",
        "category": "Recyclable (metal)",
        "bin": "Dry recyclable waste",
        "tip": "Rinse food cans and empty aerosol cans completely before recycling.",
    },
    "paper": {
        "icon": "📄",
        "display_name": "Paper",
        "category": "Recyclable (paper fibre)",
        "bin": "Dry recyclable waste",
        "tip": "Keep paper clean and dry. Food-soiled or wax-coated paper is usually not recyclable.",
    },
    "plastic": {
        "icon": "🧴",
        "display_name": "Plastic",
        "category": "Recyclable (depends on type)",
        "bin": "Dry recyclable waste",
        "tip": "Rinse containers and check the resin code and local rules, because not every plastic is accepted.",
    },
    "shoes": {
        "icon": "👟",
        "display_name": "Shoes / footwear",
        "category": "Reuse / donate",
        "bin": "Donation centre or footwear take-back",
        "tip": "Wearable shoes can be donated. Worn-out pairs may go to a specialised recycler.",
    },
    "trash": {
        "icon": "🗑️",
        "display_name": "General waste (trash)",
        "category": "Non-recyclable (residual)",
        "bin": "General / residual waste bin",
        "tip": "Check first whether any part can be recycled. Only true residual waste belongs here.",
    },
    "white-glass": {
        "icon": "🥛",
        "display_name": "White (clear) glass",
        "category": "Recyclable (glass)",
        "bin": "Dry recyclable waste / glass collection",
        "tip": "Empty and rinse the container. Wrap broken glass safely before disposal.",
    },
}

DISCLAIMER = (
    "General guidance only. Waste rules differ by city and municipality, "
    "so always follow your local authority's instructions."
)


def get_info(class_name: str) -> dict:
    """Return guidance for a class (case-insensitive), with a safe fallback for unknown names."""
    key = class_name.strip().lower().replace(" ", "-")
    return CLASS_INFO.get(
        key,
        {
            "icon": "♻️",
            "display_name": class_name,
            "category": "Unknown",
            "bin": "Check local guidelines",
            "tip": "No guidance available for this class.",
        },
    )