"""Spanish display names and management recommendations per class.

Keyed by the raw folder/class name produced during training (see
models/class_names.json). Add an entry here whenever a new class is
introduced in the dataset.
"""

RECOMMENDATIONS = {
    "Healthy": {
        "display_name": "Sana",
        "recommendation": (
            "La planta no muestra signos visibles de enfermedad. Mantenga el riego, "
            "la fertilizacion y el monitoreo periodico habituales para conservar su estado."
        ),
    },
    "Disease": {
        "display_name": "Enferma",
        "recommendation": (
            "Se detectaron signos de enfermedad o estres en la hoja/fruto. Aisle la "
            "muestra, revise el resto del cultivo en busca de sintomas similares y "
            "consulte a un especialista agronomo para identificar el agente especifico "
            "y definir el tratamiento adecuado (fungicida, manejo cultural, etc.)."
        ),
    },
}


def get_display_name(class_name: str) -> str:
    return RECOMMENDATIONS.get(class_name, {}).get("display_name", class_name)


def get_recommendation(class_name: str) -> str:
    return RECOMMENDATIONS.get(class_name, {}).get(
        "recommendation",
        "No hay una recomendacion especifica disponible para esta clase.",
    )
