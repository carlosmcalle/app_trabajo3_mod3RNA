import streamlit as st
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# --- CARGA DE MODELO Y DATOS ---
@st.cache_resource
def load_recommender_artifacts():
    model_path = Path(__file__).parents[1] / "models" / "model_module_3.pkl"
    if not model_path.exists():
        st.error("❌ No se encontró el archivo del modelo en la carpeta `models/`.")
        return None
    try:
        artifacts = joblib.load(model_path)
        return artifacts
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None

artifacts = load_recommender_artifacts()
if artifacts is None:
    st.stop()

model = artifacts.get("model")
dataset = artifacts.get("dataset")
df_dest = artifacts.get("df_dest")
df_users = artifacts.get("df_users")
df_interactions = artifacts.get("df_interactions_combined")

if any(x is None for x in [model, dataset, df_dest, df_users, df_interactions]):
    st.error("❌ Faltan componentes en el archivo del modelo.")
    st.stop()

# --- INTERFAZ DE USUARIO ---
st.title("🌍 Sistema de Recomendaciones de Destinos Turísticos")
st.markdown("""
Recomendador personalizado de destinos turísticos basado en tus preferencias e historial.
Selecciona tu usuario para recibir sugerencias exclusivas.
""")

user_ids = df_users["UserID"].sort_values().unique()
selected_user = st.selectbox("Selecciona tu UserID", user_ids)

num_recs = st.selectbox("¿Cuántas recomendaciones mostrar?", [3, 5, 10], index=1)

if selected_user:
    user_profile = df_users[df_users["UserID"] == selected_user].iloc[0]
    st.subheader("👤 Perfil del Usuario")
    st.write({
        "Género": user_profile.get("Gender", "N/A"),
        "Preferencia 1": user_profile.get("Preference_1", "N/A"),
        "Preferencia 2": user_profile.get("Preference_2", "N/A"),
        "Adultos": user_profile.get("NumberOfAdults", "N/A"),
        "Niños": user_profile.get("NumberOfChildren", "N/A"),
    })

    st.subheader("🗂️ Historial de Interacciones Positivas (Rating ≥ 4)")
    pos_hist = df_interactions[
        (df_interactions["UserID"] == selected_user) &
        (df_interactions["ExperienceRating"] >= 4)
    ]
    if not pos_hist.empty:
        id_to_name = df_dest.set_index("DestinationID")["Name"].to_dict()
        hist_table = pos_hist[["DestinationID", "ExperienceRating"]].copy()
        hist_table["Destino"] = hist_table["DestinationID"].map(id_to_name)
        st.dataframe(hist_table[["Destino", "DestinationID", "ExperienceRating"]].reset_index(drop=True))
    else:
        st.warning("El usuario no tiene interacciones positivas registradas.")

    st.subheader("✨ Recomendaciones Personalizadas")
    try:
        user_id_map, _, item_id_map, _ = dataset.mapping()
        item_id_map_rev = {v: k for k, v in item_id_map.items()}
        id_to_name = df_dest.set_index("DestinationID")["Name"].to_dict()

        if selected_user in user_id_map:
            internal_user_id = user_id_map[selected_user]
            all_item_indices = np.arange(len(item_id_map))

            visited_items = df_interactions[df_interactions["UserID"] == selected_user]["DestinationID"].values
            internal_visited_indices = [item_id_map[item] for item in visited_items if item in item_id_map]

            # Predecir scores para todos los destinos
            scores = model.predict(
                internal_user_id, all_item_indices,
                user_features=artifacts.get("user_features_matrix1"),
                item_features=artifacts.get("item_features_matrix1")
            )
            scores[internal_visited_indices] = -np.inf
            top_items_indices = np.argsort(-scores)[:num_recs]

            recs = []
            for item_index in top_items_indices:
                original_item_id = item_id_map_rev.get(item_index)
                if original_item_id:
                    recs.append({
                        "Destino": id_to_name.get(original_item_id, "Desconocido"),
                        "ID": original_item_id,
                        "Score": scores[item_index]
                    })
            if recs:
                for i, rec in enumerate(recs, 1):
                    st.success(f"{i}. {rec['Destino']} (ID: {rec['ID']})")
                with st.expander("Ver detalles técnicos"):
                    st.dataframe(pd.DataFrame(recs))
            else:
                st.warning("No se pudieron generar recomendaciones para este usuario.")
        else:
            st.warning("Usuario no encontrado en el modelo.")
    except Exception as e:
        st.error(f"Error generando recomendaciones: {e}")
