"""
Système de Prédiction d'Approbation de Carte de Crédit - Tableau de Bord Streamlit

Cette application fournit une interface interactive pour :
- Explorer le jeu de données de cartes de crédit
- Visualiser les métriques de performance du modèle
- Effectuer des prédictions d'approbation de carte de crédit
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

# Importer les modules personnalisés
from src.preprocessing import DataPreprocessor
from src.model import CreditCardModel
from src.evaluation import calculate_metrics, generate_confusion_matrix, plot_confusion_matrix, plot_feature_importance
from src.validation import validate_prediction_input

# CSS personnalisé pour un design moderne
def load_custom_css():
    """Charge le CSS personnalisé pour styliser l'application."""
    st.markdown("""
    <style>
    /* Style général */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Cartes stylisées */
    .stMetric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        transition: transform 0.3s ease;
    }
    
    .stMetric:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(0,0,0,0.3);
    }
    
    /* Boutons stylisés */
    .stButton>button {
        background: linear-gradient(135deg, #FF6B6B 0%, #FF8E53 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 15px 30px;
        font-size: 16px;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(255, 107, 107, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 107, 107, 0.6);
    }
    
    /* En-têtes stylisés */
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        text-align: center;
        padding: 20px 0;
    }
    
    h2, h3 {
        color: #667eea;
        font-weight: 700;
    }
    
    /* Sidebar stylisée */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Cartes d'information */
    .stAlert {
        border-radius: 15px;
        border-left: 5px solid #667eea;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    /* Dataframes stylisés */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    /* Inputs stylisés */
    .stNumberInput>div>div>input,
    .stSelectbox>div>div>select {
        border-radius: 10px;
        border: 2px solid #667eea;
        padding: 10px;
        transition: all 0.3s ease;
    }
    
    .stNumberInput>div>div>input:focus,
    .stSelectbox>div>div>select:focus {
        border-color: #FF6B6B;
        box-shadow: 0 0 10px rgba(255, 107, 107, 0.3);
    }
    
    /* Animation de chargement */
    .stSpinner > div {
        border-top-color: #667eea !important;
    }
    
    /* Graphiques */
    .js-plotly-plot {
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)


def main():
    """Point d'entrée principal de l'application"""
    # Charger le CSS personnalisé
    load_custom_css()
    
    # Configurer la page
    st.set_page_config(
        page_title="Prédiction d'Approbation de Carte de Crédit",
        page_icon="💳",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Titre et description avec style
    st.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <h1 style='font-size: 3em; margin-bottom: 10px;'>💳 Système de Prédiction</h1>
        <h2 style='color: #667eea; font-size: 1.5em;'>Approbation de Carte de Crédit</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align: center; padding: 10px; margin-bottom: 30px;'>
        <p style='font-size: 1.1em; color: #888;'>
        Cette application utilise l'apprentissage automatique pour prédire les décisions d'approbation de carte de crédit.
        Naviguez à travers les pages en utilisant la barre latérale pour explorer les données, visualiser les performances 
        du modèle ou effectuer des prédictions.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation dans la barre latérale
    st.sidebar.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <h2 style='color: white;'>🧭 Navigation</h2>
    </div>
    """, unsafe_allow_html=True)
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Sélectionnez une page :",
        ["📊 Explorateur de Données", "📈 Performance du Modèle", "🔮 Faire une Prédiction"],
        index=0
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **À propos de cette application :**
    
    Ce système utilise un classificateur Decision Tree pour prédire l'approbation de carte de crédit 
    basé sur les informations du demandeur incluant le revenu, l'âge, l'historique de crédit, et plus encore.
    
    **Précision du modèle : 97.35%**
    """)
    
    # Router vers la page appropriée
    if page == "📊 Explorateur de Données":
        show_data_explorer()
    elif page == "📈 Performance du Modèle":
        show_model_performance()
    else:
        show_prediction_interface()


def show_data_explorer():
    """Afficher la page Explorateur de Données avec des visualisations complètes."""
    st.header("📊 Explorateur de Données")
    st.markdown("Explorez le jeu de données de cartes de crédit avec des visualisations interactives.")
    
    # Charger les données
    data = load_dataset()
    
    if data is not None:
        st.success(f"✅ Jeu de données chargé avec succès ! Dimensions : {data.shape}")
        
        # Afficher les informations de base
        st.subheader("📋 Vue d'ensemble du jeu de données")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📝 Total d'enregistrements", data.shape[0])
        with col2:
            st.metric("🔢 Total de caractéristiques", data.shape[1])
        with col3:
            st.metric("❓ Valeurs manquantes", data.isnull().sum().sum())
        with col4:
            # Compter la distribution de la variable cible
            target_counts = data['card'].value_counts()
            approval_rate = (target_counts.get('yes', 0) / len(data)) * 100
            st.metric("✅ Taux d'approbation", f"{approval_rate:.1f}%")
        
        # Afficher un échantillon de données
        st.subheader("🔍 Échantillon de données")
        st.dataframe(data.head(10), use_container_width=True)
        
        # Afficher les statistiques récapitulatives
        st.subheader("📊 Statistiques descriptives")
        st.dataframe(data.describe(), use_container_width=True)
        
        # Visualisations de distribution pour les caractéristiques numériques
        st.subheader("📈 Distribution des caractéristiques")
        st.markdown("Histogrammes interactifs montrant la distribution des caractéristiques numériques.")
        
        # Obtenir les colonnes numériques (en excluant les catégorielles)
        numerical_cols = [col for col in data.columns 
                         if col not in ['card', 'owner', 'selfemp']]
        
        # Créer des graphiques de distribution dans une grille
        num_cols = 3
        num_rows = (len(numerical_cols) + num_cols - 1) // num_cols
        
        for i in range(0, len(numerical_cols), num_cols):
            cols = st.columns(num_cols)
            for j, col_name in enumerate(numerical_cols[i:i+num_cols]):
                with cols[j]:
                    fig = px.histogram(
                        data, 
                        x=col_name,
                        nbins=30,
                        title=f"Distribution de {col_name}",
                        color_discrete_sequence=['#667eea']
                    )
                    fig.update_layout(
                        showlegend=False,
                        height=300,
                        margin=dict(l=20, r=20, t=40, b=20),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        # Carte de chaleur de corrélation
        st.subheader("🔥 Carte de chaleur des corrélations")
        st.markdown("Matrice de corrélation montrant les relations entre les caractéristiques numériques.")
        
        # Calculer la matrice de corrélation pour les caractéristiques numériques
        numerical_data = data[numerical_cols]
        correlation_matrix = numerical_data.corr()
        
        # Créer une carte de chaleur avec plotly
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=correlation_matrix.values.round(2),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Corrélation")
        ))
        
        fig.update_layout(
            title="Matrice de corrélation des caractéristiques",
            xaxis_title="Caractéristiques",
            yaxis_title="Caractéristiques",
            height=600,
            width=800,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Distribution de la variable cible
        st.subheader("🎯 Distribution de la variable cible")
        st.markdown("Distribution des décisions d'approbation de carte de crédit.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Graphique circulaire
            target_counts = data['card'].value_counts()
            fig = px.pie(
                values=target_counts.values,
                names=['Rejeté' if x == 'no' else 'Approuvé' for x in target_counts.index],
                title="Distribution des approbations de carte de crédit",
                color_discrete_sequence=['#FF6B6B', '#4ECDC4']
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Graphique à barres
            fig = px.bar(
                x=['Rejeté' if x == 'no' else 'Approuvé' for x in target_counts.index],
                y=target_counts.values,
                title="Nombre d'approbations de carte de crédit",
                labels={'x': 'Statut d\'approbation', 'y': 'Nombre'},
                color=['Rejeté' if x == 'no' else 'Approuvé' for x in target_counts.index],
                color_discrete_sequence=['#FF6B6B', '#4ECDC4']
            )
            fig.update_layout(
                showlegend=False,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Insights supplémentaires
        st.subheader("💡 Insights clés")
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"""
            **Caractéristiques du jeu de données :**
            - Total de demandes : {len(data)}
            - Approuvées : {target_counts.get('yes', 0)} ({approval_rate:.1f}%)
            - Rejetées : {target_counts.get('no', 0)} ({100-approval_rate:.1f}%)
            """)
        
        with col2:
            # Calculer quelques statistiques
            avg_age = data['age'].mean()
            avg_income = data['income'].mean()
            st.info(f"""
            **Profil du demandeur :**
            - Âge moyen : {avg_age:.1f} ans
            - Revenu moyen : {avg_income*10000:.0f}$
            - Dépense moyenne : {data['expenditure'].mean():.0f}$
            """)
    else:
        st.error("❌ Échec du chargement du jeu de données. Veuillez vous assurer que le fichier de données existe dans le répertoire data/.")


def show_model_performance():
    """Afficher la page Performance du Modèle avec métriques et visualisations."""
    st.header("📈 Performance du Modèle")
    st.markdown("Visualisez les métriques détaillées et les visualisations de la performance du modèle.")
    
    # Charger le modèle et les données
    model = load_model()
    data = load_dataset()
    
    if model is None:
        st.warning("""
        ⚠️ Aucun modèle entraîné trouvé. 
        
        Veuillez d'abord exécuter le script d'entraînement :
        ```bash
        python train_model.py
        ```
        """)
        return
    
    if data is None:
        st.error("❌ Échec du chargement du jeu de données.")
        return
    
    st.success("✅ Modèle chargé avec succès !")
    
    # Prétraiter les données et faire des prédictions
    with st.spinner("⏳ Évaluation de la performance du modèle..."):
        # Prétraiter les données
        preprocessor = DataPreprocessor("data/AER_credit_card_data.csv")
        df = data.copy()
        
        # Appliquer les étapes de prétraitement
        df = preprocessor.handle_missing_values(df)
        df = preprocessor.handle_outliers(df)
        df = preprocessor.encode_categorical_features(df)
        
        # Diviser les données
        X_train, X_test, y_train, y_test = preprocessor.split_data(df, test_size=0.2, random_state=42)
        
        # Mettre à l'échelle les caractéristiques
        X_train_scaled = preprocessor.scale_numerical_features(X_train)
        X_test_scaled = preprocessor.scale_numerical_features(X_test)
        
        # Faire des prédictions
        y_test_pred = model.predict(X_test_scaled.values)
        y_test_proba = model.predict_proba(X_test_scaled.values)
        
        # Calculer les métriques
        metrics = calculate_metrics(y_test.values, y_test_pred)
        cm = generate_confusion_matrix(y_test.values, y_test_pred)
    
    # Afficher les métriques dans des colonnes
    st.subheader("📊 Métriques de performance")
    st.markdown("Métriques clés évaluant la précision des prédictions du modèle.")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="🎯 Exactitude",
            value=f"{metrics['accuracy']:.2%}",
            help="Justesse globale des prédictions"
        )
    
    with col2:
        st.metric(
            label="🔍 Précision",
            value=f"{metrics['precision']:.2%}",
            help="Exactitude des prédictions positives"
        )
    
    with col3:
        st.metric(
            label="📡 Rappel",
            value=f"{metrics['recall']:.2%}",
            help="Capacité à trouver tous les cas positifs"
        )
    
    with col4:
        st.metric(
            label="⚖️ Score F1",
            value=f"{metrics['f1_score']:.2%}",
            help="Moyenne harmonique de la précision et du rappel"
        )
    
    # Ajouter une interprétation
    st.info("""
    **Interprétation des métriques :**
    - **Exactitude** : Pourcentage de prédictions correctes globalement
    - **Précision** : Parmi toutes les approbations prédites, combien étaient réellement approuvées
    - **Rappel** : Parmi toutes les approbations réelles, combien avons-nous correctement prédit
    - **Score F1** : Équilibre entre la précision et le rappel
    """)
    
    # Afficher la matrice de confusion
    st.subheader("🔲 Matrice de confusion")
    st.markdown("Représentation visuelle de la précision des prédictions par classe.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Créer une visualisation de la matrice de confusion
        fig = plot_confusion_matrix(cm, class_names=['Non', 'Oui'])
        st.pyplot(fig)
    
    with col2:
        st.markdown("### 📋 Détails de la matrice")
        st.markdown(f"""
        **Vrais Négatifs (VN) :** {cm[0, 0]}  
        *Rejets correctement prédits*
        
        **Faux Positifs (FP) :** {cm[0, 1]}  
        *Approbations incorrectement prédites*
        
        **Faux Négatifs (FN) :** {cm[1, 0]}  
        *Rejets incorrectement prédits*
        
        **Vrais Positifs (VP) :** {cm[1, 1]}  
        *Approbations correctement prédites*
        """)
        
        # Calculer des métriques supplémentaires
        total = cm.sum()
        correct = cm[0, 0] + cm[1, 1]
        st.success(f"**{correct}/{total}** prédictions correctes")
    
    # Afficher l'importance des caractéristiques
    st.subheader("⭐ Importance des caractéristiques")
    st.markdown("Importance relative de chaque caractéristique dans les prédictions.")
    
    # Obtenir l'importance des caractéristiques
    feature_importance = model.get_feature_importance()
    
    # Trier par importance
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    feature_names = [f[0] for f in sorted_features]
    importance_values = [f[1] for f in sorted_features]
    
    # Créer un graphique à barres horizontales avec plotly
    fig = px.bar(
        x=importance_values,
        y=feature_names,
        orientation='h',
        title="Scores d'importance des caractéristiques",
        labels={'x': 'Score d\'importance', 'y': 'Caractéristique'},
        color=importance_values,
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        showlegend=False,
        height=400,
        yaxis={'categoryorder': 'total ascending'},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Afficher les principales caractéristiques
    st.markdown("### 🏆 Top 5 des caractéristiques les plus importantes")
    for i, (feature, importance) in enumerate(sorted_features[:5], 1):
        st.write(f"{i}. **{feature}** : {importance:.4f}")
    
    # Informations sur le modèle
    st.subheader("ℹ️ Informations sur le modèle")
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"""
        **Type de modèle :** Classificateur Decision Tree
        
        **Taille de l'ensemble d'entraînement :** {len(X_train)} échantillons
        
        **Taille de l'ensemble de test :** {len(X_test)} échantillons
        
        **Nombre de caractéristiques :** {X_test.shape[1]}
        """)
    
    with col2:
        if model.best_params:
            st.info(f"""
            **Meilleurs hyperparamètres :**
            
            - Critère : {model.best_params.get('criterion', 'N/A')}
            - Profondeur max : {model.best_params.get('max_depth', 'N/A')}
            - Min échantillons division : {model.best_params.get('min_samples_split', 'N/A')}
            - Min échantillons feuille : {model.best_params.get('min_samples_leaf', 'N/A')}
            """)
        else:
            st.info("Informations sur les hyperparamètres non disponibles.")


def show_prediction_interface():
    """Afficher la page Faire une Prédiction avec champs de saisie et logique de prédiction."""
    st.header("🔮 Faire une Prédiction")
    st.markdown("Entrez les informations du demandeur pour prédire l'approbation de la carte de crédit.")
    
    # Charger le modèle
    model = load_model()
    
    if model is None:
        st.warning("""
        ⚠️ Aucun modèle entraîné trouvé. 
        
        Veuillez d'abord exécuter le script d'entraînement :
        ```bash
        python train_model.py
        ```
        """)
        return
    
    st.success("✅ Modèle chargé avec succès !")
    
    # Créer le formulaire de saisie
    st.subheader("📝 Informations du demandeur")
    st.markdown("Veuillez remplir tous les champs requis ci-dessous.")
    
    # Créer deux colonnes pour les champs de saisie
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 👤 Informations personnelles")
        
        age = st.number_input(
            "Âge (années + fraction)",
            min_value=18.0,
            max_value=100.0,
            value=30.0,
            step=0.1,
            help="Âge en années avec fraction décimale"
        )
        
        owner = st.selectbox(
            "Propriétaire de maison",
            options=["oui", "non"],
            help="Le demandeur est-il propriétaire de sa maison ?"
        )
        
        selfemp = st.selectbox(
            "Travailleur autonome",
            options=["oui", "non"],
            help="Le demandeur est-il travailleur autonome ?"
        )
        
        dependents = st.number_input(
            "Nombre de personnes à charge",
            min_value=0,
            max_value=20,
            value=0,
            step=1,
            help="Nombre de personnes à charge"
        )
        
        months = st.number_input(
            "Mois à l'adresse actuelle",
            min_value=0,
            max_value=600,
            value=12,
            step=1,
            help="Nombre de mois vivant à l'adresse actuelle"
        )
    
    with col2:
        st.markdown("### 💰 Informations financières")
        
        income = st.number_input(
            "Revenu annuel (en 10 000$)",
            min_value=0.0,
            max_value=100.0,
            value=5.0,
            step=0.1,
            help="Revenu annuel divisé par 10 000 (ex: 5.0 = 50 000$)"
        )
        
        expenditure = st.number_input(
            "Dépenses mensuelles de carte de crédit ($)",
            min_value=0.0,
            max_value=10000.0,
            value=100.0,
            step=10.0,
            help="Dépenses mensuelles moyennes de carte de crédit"
        )
        
        share = st.number_input(
            "Ratio dépenses/revenu",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.01,
            help="Ratio des dépenses mensuelles de carte de crédit sur le revenu"
        )
        
        reports = st.number_input(
            "Rapports dérogatoires majeurs",
            min_value=0,
            max_value=20,
            value=0,
            step=1,
            help="Nombre de rapports dérogatoires majeurs"
        )
        
        majorcards = st.number_input(
            "Nombre de cartes de crédit majeures",
            min_value=0,
            max_value=20,
            value=1,
            step=1,
            help="Nombre de cartes de crédit majeures détenues"
        )
        
        active = st.number_input(
            "Comptes de crédit actifs",
            min_value=0,
            max_value=50,
            value=1,
            step=1,
            help="Nombre de comptes de crédit actifs"
        )
    
    # Bouton de prédiction
    st.markdown("---")
    
    if st.button("🔮 Prédire l'approbation de la carte de crédit", type="primary", use_container_width=True):
        # Convertir les valeurs françaises en anglais pour le modèle
        owner_en = "yes" if owner == "oui" else "no"
        selfemp_en = "yes" if selfemp == "oui" else "no"
        
        # Valider les entrées
        validation_errors = validate_prediction_input({
            'age': age,
            'owner': owner_en,
            'selfemp': selfemp_en,
            'dependents': dependents,
            'months': months,
            'income': income,
            'expenditure': expenditure,
            'share': share,
            'reports': reports,
            'majorcards': majorcards,
            'active': active
        })
        
        if validation_errors:
            st.error("❌ Erreurs de validation :")
            for error in validation_errors:
                st.error(f"  • {error}")
        else:
            # Faire la prédiction
            with st.spinner("⏳ Prédiction en cours..."):
                # Créer le dataframe d'entrée
                input_data = pd.DataFrame({
                    'reports': [reports],
                    'age': [age],
                    'income': [income],
                    'share': [share],
                    'expenditure': [expenditure],
                    'owner': [owner_en],
                    'selfemp': [selfemp_en],
                    'dependents': [dependents],
                    'months': [months],
                    'majorcards': [majorcards],
                    'active': [active]
                })
                
                # Prétraiter l'entrée
                preprocessor = DataPreprocessor("data/AER_credit_card_data.csv")
                
                # Encoder les caractéristiques catégorielles
                input_data = preprocessor.encode_categorical_features(input_data)
                
                # Mettre à l'échelle les caractéristiques numériques
                input_data = preprocessor.scale_numerical_features(input_data)
                
                # Faire la prédiction
                prediction = model.predict(input_data.values)[0]
                probability = model.predict_proba(input_data.values)[0]
                
                # Obtenir la probabilité pour la classe 'yes'
                # Le modèle retourne les probabilités pour ['no', 'yes']
                prob_yes = probability[1] if len(probability) > 1 else probability[0]
                
                # Déterminer le niveau de confiance
                if prob_yes > 0.8 or prob_yes < 0.2:
                    confidence = "Élevée"
                    confidence_color = "green"
                elif 0.6 < prob_yes < 0.8 or 0.2 < prob_yes < 0.4:
                    confidence = "Moyenne"
                    confidence_color = "orange"
                else:
                    confidence = "Faible"
                    confidence_color = "red"
            
            # Afficher les résultats
            st.markdown("---")
            st.subheader("📊 Résultats de la prédiction")
            
            # Résultat principal de la prédiction
            if prediction == 'yes':
                st.markdown("""
                <div style='background: linear-gradient(135deg, #4ECDC4 0%, #44A08D 100%); 
                            padding: 30px; border-radius: 20px; text-align: center; 
                            box-shadow: 0 10px 30px rgba(78, 205, 196, 0.3);'>
                    <h2 style='color: white; font-size: 2.5em; margin: 0;'>✅ APPROUVÉ</h2>
                    <p style='color: white; font-size: 1.2em; margin-top: 10px;'>
                        Demande de carte de crédit approuvée !
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style='background: linear-gradient(135deg, #FF6B6B 0%, #C44569 100%); 
                            padding: 30px; border-radius: 20px; text-align: center; 
                            box-shadow: 0 10px 30px rgba(255, 107, 107, 0.3);'>
                    <h2 style='color: white; font-size: 2.5em; margin: 0;'>❌ REFUSÉ</h2>
                    <p style='color: white; font-size: 1.2em; margin-top: 10px;'>
                        Demande de carte de crédit refusée
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Afficher la probabilité et la confiance
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="📈 Probabilité d'approbation",
                    value=f"{prob_yes:.1%}",
                    help="Probabilité que la demande soit approuvée"
                )
            
            with col2:
                st.metric(
                    label="📉 Probabilité de refus",
                    value=f"{1-prob_yes:.1%}",
                    help="Probabilité que la demande soit refusée"
                )
            
            with col3:
                st.metric(
                    label="🎯 Niveau de confiance",
                    value=confidence,
                    help="Confiance du modèle dans la prédiction"
                )
            
            # Barre de probabilité
            st.markdown("### 📊 Distribution des probabilités")
            prob_df = pd.DataFrame({
                'Décision': ['Refusé', 'Approuvé'],
                'Probabilité': [1-prob_yes, prob_yes]
            })
            
            fig = px.bar(
                prob_df,
                x='Décision',
                y='Probabilité',
                color='Décision',
                color_discrete_map={'Refusé': '#FF6B6B', 'Approuvé': '#4ECDC4'},
                text='Probabilité'
            )
            
            fig.update_traces(texttemplate='%{text:.1%}', textposition='outside')
            fig.update_layout(
                showlegend=False,
                yaxis_range=[0, 1],
                height=400,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Informations supplémentaires
            st.info(f"""
            **💡 Interprétation :**
            
            Le modèle prédit que cette demande sera **{prediction.upper()}** avec une 
            probabilité de **{prob_yes:.1%}**. Le niveau de confiance est **{confidence}**, 
            indiquant une certitude {'forte' if confidence == 'Élevée' else 'modérée' if confidence == 'Moyenne' else 'faible'} 
            dans cette prédiction.
            """)


@st.cache_data
def load_dataset():
    """
    Charger le jeu de données de cartes de crédit avec mise en cache.
    
    Returns:
        pd.DataFrame: Le jeu de données chargé, ou None en cas d'échec
    """
    try:
        data_path = Path("data/AER_credit_card_data.csv")
        if not data_path.exists():
            st.error(f"Jeu de données introuvable à {data_path}")
            return None
        
        data = pd.read_csv(data_path)
        return data
    except Exception as e:
        st.error(f"Erreur lors du chargement du jeu de données : {str(e)}")
        return None


@st.cache_resource
def load_model():
    """
    Charger le modèle entraîné avec mise en cache.
    
    Returns:
        CreditCardModel: Le modèle chargé, ou None en cas d'échec
    """
    try:
        model_path = Path("models/credit_card_model.pkl")
        if not model_path.exists():
            return None
        
        model = CreditCardModel()
        model.load_model(str(model_path))
        return model
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {str(e)}")
        return None


if __name__ == "__main__":
    main()
