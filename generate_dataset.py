#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script de génération de dataset pour le projet de prédiction du revenu annuel des Marocains.
Auteur: [Votre nom]
Date: Mai 2025

Ce script génère un dataset de revenus marocains selon les contraintes statistiques
spécifiées par le HCP (Haut-Commissariat au Plan).
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import random
from datetime import datetime, timedelta

# Configuration de la seed pour la reproductibilité
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# Configuration
NB_RECORDS = 40000
OUTPUT_FILE = "dataset_revenu_marocains.csv"

# Statistiques selon le HCP
REVENU_MOYEN_GLOBAL = 21949
REVENU_MOYEN_URBAIN = 26988
REVENU_MOYEN_RURAL = 12862

# Pourcentage de la population en dessous de la moyenne
PCT_BELOW_MEAN_GLOBAL = 0.718
PCT_BELOW_MEAN_URBAIN = 0.659
PCT_BELOW_MEAN_RURAL = 0.854

# Répartition urbain/rural et hommes/femmes au Maroc (approximatif)
PCT_URBAIN = 0.63
PCT_HOMMES = 0.50

def generate_dataset():
    """
    Génère le dataset complet des revenus marocains selon les contraintes spécifiées.
    
    Returns:
        pandas.DataFrame: Le dataset généré
    """
    print("Génération du dataset en cours...")
    
    # 1. Création de la structure de base
    df = pd.DataFrame()
    
    # 2. Génération des variables démographiques
    generate_demographic_variables(df)
    
    # 3. Génération des variables socio-économiques
    generate_socioeconomic_variables(df)
    
    # 4. Génération des variables supplémentaires personnalisées
    generate_custom_variables(df)
    
    # 5. Génération du revenu en fonction des variables
    generate_income(df)
    
    # 6. Ajout de valeurs manquantes, aberrantes et colonnes redondantes/non pertinentes
    add_data_quality_issues(df)
    
    print(f"Dataset généré avec succès: {len(df)} enregistrements et {len(df.columns)} colonnes")
    return df

def generate_demographic_variables(df):
    """
    Génère les variables démographiques: âge, sexe, milieu (urbain/rural),
    état civil, catégorie d'âge.
    
    Args:
        df: DataFrame à remplir
    """
    # Milieu (urbain/rural)
    df['milieu'] = np.random.choice(['urbain', 'rural'], size=NB_RECORDS, 
                                   p=[PCT_URBAIN, 1-PCT_URBAIN])
    
    # Sexe
    df['sexe'] = np.random.choice(['homme', 'femme'], size=NB_RECORDS, 
                                 p=[PCT_HOMMES, 1-PCT_HOMMES])
    
    # Âge - distribution qui reflète la pyramide des âges marocaine
    # Plus de jeunes, moins de personnes âgées
    age_mean = 30
    age_std = 15
    df['age'] = np.random.normal(age_mean, age_std, NB_RECORDS).astype(int)
    # Limiter l'âge entre 18 et 85 ans
    df['age'] = np.clip(df['age'], 18, 85)
    
    # Catégorie d'âge
    conditions = [
        (df['age'] < 30),
        (df['age'] >= 30) & (df['age'] < 50),
        (df['age'] >= 50) & (df['age'] < 65),
        (df['age'] >= 65)
    ]
    categories = ['jeune', 'adulte', 'senior', 'age']
    df['categorie_age'] = np.select(conditions, categories, default='adulte')
    
    # État matrimonial - dépend de l'âge
    df['etat_matrimonial'] = 'celibataire'  # Valeur par défaut
    
    # Probabilités différentes selon l'âge
    for idx, row in df.iterrows():
        age = row['age']
        if age < 25:
            etat = np.random.choice(['celibataire', 'marie'], p=[0.9, 0.1])
        elif age < 35:
            etat = np.random.choice(['celibataire', 'marie', 'divorce'], p=[0.4, 0.55, 0.05])
        elif age < 50:
            etat = np.random.choice(['celibataire', 'marie', 'divorce', 'veuf'], p=[0.2, 0.65, 0.1, 0.05])
        else:
            etat = np.random.choice(['celibataire', 'marie', 'divorce', 'veuf'], p=[0.1, 0.6, 0.15, 0.15])
        df.at[idx, 'etat_matrimonial'] = etat

def generate_socioeconomic_variables(df):
    """
    Génère les variables socio-économiques: niveau d'éducation, catégorie
    socioprofessionnelle, années d'expérience, possession de biens.
    
    Args:
        df: DataFrame à remplir
    """
    # Répartition niveau d'éducation - dépend du milieu
    for idx, row in df.iterrows():
        if row['milieu'] == 'urbain':
            education = np.random.choice(
                ['sans_niveau', 'fondamental', 'secondaire', 'superieur'],
                p=[0.15, 0.30, 0.35, 0.20]
            )
        else:  # Rural
            education = np.random.choice(
                ['sans_niveau', 'fondamental', 'secondaire', 'superieur'],
                p=[0.40, 0.40, 0.15, 0.05]
            )
        df.at[idx, 'niveau_education'] = education
    
    # Années d'expérience - dépend de l'âge et du niveau d'éducation
    df['annees_experience'] = 0  # Valeur par défaut
    
    for idx, row in df.iterrows():
        age = row['age']
        education = row['niveau_education']
        
        if education == 'sans_niveau':
            max_exp = max(0, age - 15)  # Commencent à travailler tôt
        elif education == 'fondamental':
            max_exp = max(0, age - 18)
        elif education == 'secondaire':
            max_exp = max(0, age - 20)
        else:  # supérieur
            max_exp = max(0, age - 24)
        
        # L'expérience ne peut pas dépasser l'âge moins l'âge de début de travail
        # Et on ajoute un peu d'aléatoire
        exp = int(min(max_exp, max_exp * 0.8 + max_exp * 0.2 * np.random.random()))
        df.at[idx, 'annees_experience'] = exp
    
    # Catégorie socioprofessionnelle - dépend du niveau d'éducation, milieu et expérience
    df['categorie_sociopro'] = 'groupe_6'  # Valeur par défaut
    
    for idx, row in df.iterrows():
        education = row['niveau_education']
        milieu = row['milieu']
        exp = row['annees_experience']
        age = row['age']
        
        # Probabilités de base pour chaque groupe
        if education == 'superieur':
            probs = [0.25, 0.35, 0.10, 0.05, 0.15, 0.10]
        elif education == 'secondaire':
            probs = [0.05, 0.30, 0.15, 0.10, 0.25, 0.15]
        elif education == 'fondamental':
            probs = [0.01, 0.10, 0.20, 0.25, 0.24, 0.20]
        else:  # sans niveau
            probs = [0.001, 0.02, 0.15, 0.30, 0.23, 0.299]
        
        # Ajustement selon l'expérience
        if exp > 15:
            # Plus d'expérience augmente les chances pour les groupes supérieurs
            probs = [p * 1.5 if i < 3 else p for i, p in enumerate(probs)]
        
        # Ajustement selon le milieu
        if milieu == 'rural':
            # Plus de chances d'être dans le groupe 4 (agriculteurs) en milieu rural
            probs[3] *= 2.5
        
        # Ajustement pour les retraités (groupe 3)
        if age >= 60:
            probs[2] *= 3  # Plus de chances d'être retraité
        
        # Normalisation des probabilités
        probs = [p / sum(probs) for p in probs]
        
        categorie = np.random.choice(
            ['groupe_1', 'groupe_2', 'groupe_3', 'groupe_4', 'groupe_5', 'groupe_6'],
            p=probs
        )
        df.at[idx, 'categorie_sociopro'] = categorie
    
    # Possession de biens - dépend de la catégorie socioprofessionnelle
    df['possession_voiture'] = 0
    df['possession_logement'] = 0
    df['possession_terrain'] = 0
    df['possession_investissements'] = 0
    
    for idx, row in df.iterrows():
        categorie = row['categorie_sociopro']
        
        # Probabilités selon la catégorie socioprofessionnelle
        if categorie == 'groupe_1':
            probs_voiture = 0.9
            probs_logement = 0.85
            probs_terrain = 0.6
            probs_invest = 0.7
        elif categorie == 'groupe_2':
            probs_voiture = 0.7
            probs_logement = 0.7
            probs_terrain = 0.3
            probs_invest = 0.4
        elif categorie == 'groupe_3':
            probs_voiture = 0.4
            probs_logement = 0.8
            probs_terrain = 0.3
            probs_invest = 0.35
        elif categorie == 'groupe_4':
            probs_voiture = 0.2
            probs_logement = 0.6
            probs_terrain = 0.5
            probs_invest = 0.1
        elif categorie == 'groupe_5':
            probs_voiture = 0.3
            probs_logement = 0.4
            probs_terrain = 0.15
            probs_invest = 0.1
        else:  # groupe_6
            probs_voiture = 0.1
            probs_logement = 0.2
            probs_terrain = 0.05
            probs_invest = 0.01
        
        df.at[idx, 'possession_voiture'] = 1 if np.random.random() < probs_voiture else 0
        df.at[idx, 'possession_logement'] = 1 if np.random.random() < probs_logement else 0
        df.at[idx, 'possession_terrain'] = 1 if np.random.random() < probs_terrain else 0
        df.at[idx, 'possession_investissements'] = 1 if np.random.random() < probs_invest else 0

def generate_custom_variables(df):
    """
    Génère trois variables supplémentaires personnalisées:
    1. Niveau d'endettement
    2. Nombre de personnes à charge
    3. Secteur d'activité
    
    Args:
        df: DataFrame à remplir
    """
    # 1. Niveau d'endettement (en % du revenu mensuel)
    # Dépend de l'âge et de la possession de biens
    df['niveau_endettement'] = 0.0
    
    for idx, row in df.iterrows():
        base_endettement = np.random.normal(30, 15)  # Moyenne de 30% avec écart-type de 15%
        
        # Ajustement selon l'âge
        if row['age'] < 30:
            base_endettement *= 0.7  # Les jeunes ont tendance à moins s'endetter
        elif row['age'] > 50:
            base_endettement *= 0.8  # Les seniors ont tendance à moins s'endetter
            
        # Ajustement selon les possessions
        if row['possession_logement'] == 1:
            base_endettement *= 1.3  # Crédit immobilier
        if row['possession_voiture'] == 1:
            base_endettement *= 1.2  # Crédit automobile
            
        # Limiter entre 0 et 80%
        endettement = max(0, min(80, base_endettement))
        df.at[idx, 'niveau_endettement'] = round(endettement, 2)
    
    # 2. Nombre de personnes à charge
    # Dépend de l'âge et de l'état matrimonial
    df['nb_personnes_charge'] = 0
    
    for idx, row in df.iterrows():
        if row['etat_matrimonial'] == 'celibataire':
            max_dependants = 2
            weights = [0.7, 0.2, 0.1]  # Poids pour 0, 1, 2 personnes à charge
        elif row['etat_matrimonial'] == 'marie':
            if row['age'] < 30:
                max_dependants = 3
                weights = [0.3, 0.4, 0.2, 0.1]  # Poids pour 0, 1, 2, 3 personnes
            elif row['age'] < 50:
                max_dependants = 5
                weights = [0.1, 0.2, 0.3, 0.2, 0.1, 0.1]  # Poids pour 0, 1, 2, 3, 4, 5 personnes
            else:
                max_dependants = 3
                weights = [0.4, 0.3, 0.2, 0.1]  # Poids pour 0, 1, 2, 3 personnes
        else:  # divorcé ou veuf
            max_dependants = 3
            weights = [0.3, 0.4, 0.2, 0.1]  # Poids pour 0, 1, 2, 3 personnes
            
        # Ajout d'un peu de variabilité
        dependants = np.random.choice(range(max_dependants + 1), p=weights)
        df.at[idx, 'nb_personnes_charge'] = dependants
    
    # 3. Secteur d'activité (dépend de la catégorie socioprofessionnelle et du milieu)
    # Définir les secteurs
    secteurs = [
        'agriculture', 'industrie', 'services', 'administration', 'commerce',
        'construction', 'transport', 'sante', 'education', 'tourisme', 'autre'
    ]
    
    df['secteur_activite'] = 'autre'  # Valeur par défaut
    
    for idx, row in df.iterrows():
        categorie = row['categorie_sociopro']
        milieu = row['milieu']
        
        # Définir les probabilités de base pour chaque secteur
        probs = [0.1] * len(secteurs)  # Probabilité uniforme
        
        # Ajustement selon la catégorie socioprofessionnelle
        if categorie == 'groupe_1':
            # Plus de chances dans l'administration, services, santé, éducation
            sectors_idx = [secteurs.index(s) for s in ['administration', 'services', 'sante', 'education']]
            for idx_s in sectors_idx:
                probs[idx_s] *= 2.5
        elif categorie == 'groupe_2':
            # Plus de chances dans le commerce, services, administration
            sectors_idx = [secteurs.index(s) for s in ['commerce', 'services', 'administration']]
            for idx_s in sectors_idx:
                probs[idx_s] *= 2.0
        elif categorie == 'groupe_3':
            # Retraités - répartition plus uniforme
            pass
        elif categorie == 'groupe_4':
            # Agriculture et pêche
            probs[secteurs.index('agriculture')] *= 5.0
        elif categorie == 'groupe_5':
            # Industrie, construction, transport
            sectors_idx = [secteurs.index(s) for s in ['industrie', 'construction', 'transport']]
            for idx_s in sectors_idx:
                probs[idx_s] *= 2.5
        else:  # groupe_6
            # Services, commerce, construction, autre
            sectors_idx = [secteurs.index(s) for s in ['services', 'commerce', 'construction', 'autre']]
            for idx_s in sectors_idx:
                probs[idx_s] *= 1.8
        
        # Ajustement selon le milieu
        if milieu == 'rural':
            probs[secteurs.index('agriculture')] *= 3.0
        else:  # urbain
            probs[secteurs.index('agriculture')] *= 0.2
            sectors_idx = [secteurs.index(s) for s in ['services', 'commerce', 'administration']]
            for idx_s in sectors_idx:
                probs[idx_s] *= 1.5
                
        # Normalisation des probabilités
        probs = [p / sum(probs) for p in probs]
        
        secteur = np.random.choice(secteurs, p=probs)
        df.at[idx, 'secteur_activite'] = secteur

def generate_income(df):
    """
    Génère les revenus annuels en fonction des caractéristiques.
    Respecte les contraintes statistiques du HCP.
    
    Args:
        df: DataFrame à remplir
    """
    print("Génération des revenus...")
    
    # Créer un vecteur de base pour le revenu
    df['revenu_annuel'] = 0.0
    
    # Facteurs de pondération pour chaque caractéristique
    # Ces coefficients déterminent l'impact de chaque variable sur le revenu
    weights = {
        'urbain': 1.3,            # Effet du milieu urbain
        'rural': 0.7,             # Effet du milieu rural
        'homme': 1.2,             # Effet du sexe (homme)
        'femme': 0.8,             # Effet du sexe (femme)
        'age_factor': 0.02,       # Effet de l'âge (par année)
        'exp_factor': 0.03,       # Effet de l'expérience (par année)
        
        # Effet du niveau d'éducation
        'education': {
            'sans_niveau': 0.5,
            'fondamental': 0.8,
            'secondaire': 1.2,
            'superieur': 2.0
        },
        
        # Effet de la catégorie socioprofessionnelle
        'sociopro': {
            'groupe_1': 3.0,
            'groupe_2': 2.0,
            'groupe_3': 1.5,
            'groupe_4': 1.0,
            'groupe_5': 0.8,
            'groupe_6': 0.6
        },
        
        # Effet de l'état matrimonial
        'matrimonial': {
            'celibataire': 0.9,
            'marie': 1.1,
            'divorce': 1.0,
            'veuf': 0.95
        },
        
        # Effet des possessions
        'voiture': 1.15,
        'logement': 1.2,
        'terrain': 1.25,
        'investissements': 1.3,
        
        # Autres facteurs
        'endettement_factor': -0.005,  # Effet négatif de l'endettement
        'charge_factor': -0.05,       # Effet négatif des personnes à charge
        
        # Effet du secteur d'activité
        'secteur': {
            'agriculture': 0.8,
            'industrie': 1.2,
            'services': 1.1,
            'administration': 1.3,
            'commerce': 1.1,
            'construction': 1.0,
            'transport': 1.1,
            'sante': 1.4,
            'education': 1.3,
            'tourisme': 1.0,
            'autre': 0.9
        }
    }
    
    # Calculer le revenu de base pour chaque personne
    for idx, row in df.iterrows():
        # Revenu de base selon le milieu
        if row['milieu'] == 'urbain':
            base_income = REVENU_MOYEN_URBAIN
            milieu_factor = weights['urbain']
        else:
            base_income = REVENU_MOYEN_RURAL
            milieu_factor = weights['rural']
        
        # Facteur sexe
        sexe_factor = weights[row['sexe']]
        
        # Facteur âge (croissant jusqu'à 55 ans, puis décroissant)
        age = row['age']
        age_peak = 55
        if age <= age_peak:
            age_factor = 1 + weights['age_factor'] * age
        else:
            age_factor = 1 + weights['age_factor'] * age_peak - weights['age_factor'] * 0.5 * (age - age_peak)
        
        # Facteur expérience
        exp_factor = 1 + weights['exp_factor'] * row['annees_experience']
        
        # Facteur éducation
        education_factor = weights['education'][row['niveau_education']]
        
        # Facteur catégorie socioprofessionnelle
        sociopro_factor = weights['sociopro'][row['categorie_sociopro']]
        
        # Facteur état matrimonial
        matrimonial_factor = weights['matrimonial'][row['etat_matrimonial']]
        
        # Facteur possessions
        possession_factor = 1.0
        if row['possession_voiture'] == 1:
            possession_factor *= weights['voiture']
        if row['possession_logement'] == 1:
            possession_factor *= weights['logement']
        if row['possession_terrain'] == 1:
            possession_factor *= weights['terrain']
        if row['possession_investissements'] == 1:
            possession_factor *= weights['investissements']
            
        # Facteur endettement
        endettement_factor = 1 + weights['endettement_factor'] * row['niveau_endettement']
        
        # Facteur personnes à charge
        charge_factor = 1 + weights['charge_factor'] * row['nb_personnes_charge']
        
        # Facteur secteur d'activité
        secteur_factor = weights['secteur'][row['secteur_activite']]
        
        # Calcul du revenu combinant tous les facteurs
        income = base_income * milieu_factor * sexe_factor * age_factor * exp_factor * education_factor * \
                 sociopro_factor * matrimonial_factor * possession_factor * endettement_factor * \
                 charge_factor * secteur_factor
                 
        # Ajouter un facteur aléatoire pour la variabilité
        random_factor = np.random.normal(1.0, 0.15)  # Moyenne 1.0, écart-type 0.15
        income *= random_factor
        
        df.at[idx, 'revenu_annuel'] = round(income, 2)
    
    # Ajustement pour respecter les proportions sous la moyenne
    adjust_income_distribution(df)

def adjust_income_distribution(df):
    """
    Ajuste la distribution des revenus pour respecter les contraintes du HCP:
    - 71.8% de la population globale a un revenu inférieur à la moyenne
    - 65.9% de la population urbaine a un revenu inférieur à la moyenne
    - 85.4% de la population rurale a un revenu inférieur à la moyenne
    
    Args:
        df: DataFrame à ajuster
    """
    # Vérifier les proportions actuelles
    mean_global = df['revenu_annuel'].mean()
    below_mean_global = (df['revenu_annuel'] < mean_global).mean()
    
    df_urbain = df[df['milieu'] == 'urbain']
    df_rural = df[df['milieu'] == 'rural']
    
    mean_urbain = df_urbain['revenu_annuel'].mean()
    mean_rural = df_rural['revenu_annuel'].mean()
    
    below_mean_urbain = (df_urbain['revenu_annuel'] < mean_urbain).mean()
    below_mean_rural = (df_rural['revenu_annuel'] < mean_rural).mean()
    
    print(f"Avant ajustement:")
    print(f"Proportion globale sous la moyenne: {below_mean_global:.4f} (cible: {PCT_BELOW_MEAN_GLOBAL:.4f})")
    print(f"Proportion urbaine sous la moyenne: {below_mean_urbain:.4f} (cible: {PCT_BELOW_MEAN_URBAIN:.4f})")
    print(f"Proportion rurale sous la moyenne: {below_mean_rural:.4f} (cible: {PCT_BELOW_MEAN_RURAL:.4f})")
    
    # Ajustement par transformation logistique des revenus
    for milieu, target_pct in [('urbain', PCT_BELOW_MEAN_URBAIN), ('rural', PCT_BELOW_MEAN_RURAL)]:
        df_subset = df[df['milieu'] == milieu]
        current_pct = (df_subset['revenu_annuel'] < df_subset['revenu_annuel'].mean()).mean()
        
        # Calculer le facteur d'écart
        scale_factor = np.log(target_pct / (1 - target_pct)) / np.log(current_pct / (1 - current_pct))
        
        # Appliquer la transformation
        mean_income = df_subset['revenu_annuel'].mean()
        
        for idx in df_subset.index:
            relative_income = df.at[idx, 'revenu_annuel'] / mean_income
            
            # Transformation logistique
            if relative_income < 1:
                new_relative = relative_income ** scale_factor
            else:
                new_relative = 2 - (2 - relative_income) ** scale_factor
                
            df.at[idx, 'revenu_annuel'] = mean_income * new_relative
    
    # Vérifier les nouvelles proportions
    mean_global = df['revenu_annuel'].mean()
    below_mean_global = (df['revenu_annuel'] < mean_global).mean()
    
    df_urbain = df[df['milieu'] == 'urbain']
    df_rural = df[df['milieu'] == 'rural']
    
    mean_urbain = df_urbain['revenu_annuel'].mean()
    mean_rural = df_rural['revenu_annuel'].mean()
    
    below_mean_urbain = (df_urbain['revenu_annuel'] < mean_urbain).mean()
    below_mean_rural = (df_rural['revenu_annuel'] < mean_rural).mean()
    
    print(f"Après ajustement:")
    print(f"Proportion globale sous la moyenne: {below_mean_global:.4f} (cible: {PCT_BELOW_MEAN_GLOBAL:.4f})")
    print(f"Proportion urbaine sous la moyenne: {below_mean_urbain:.4f} (cible: {PCT_BELOW_MEAN_URBAIN:.4f})")
    print(f"Proportion rurale sous la moyenne: {below_mean_rural:.4f} (cible: {PCT_BELOW_MEAN_RURAL:.4f})")
    
    # Arrondir les revenus à 2 décimales
    df['revenu_annuel'] = df['revenu_annuel'].round(2)

def add_data_quality_issues(df):
    """
    Ajoute des problèmes de qualité aux données:
    1. Valeurs manquantes
    2. Valeurs aberrantes
    3. Colonnes redondantes
    4. Colonnes non pertinentes
    
    Args:
        df: DataFrame à modifier
    """
    print("Ajout de problèmes de qualité aux données...")
    
    # 1. Valeurs manquantes (environ 5% des données)
    missing_pct = 0.05
    rows = np.random.choice(df.index, size=int(len(df) * missing_pct), replace=False)
    
    # Colonnes qui peuvent avoir des valeurs manquantes
    missing_cols = ['niveau_education', 'etat_matrimonial', 'annees_experience', 
                   'categorie_sociopro', 'niveau_endettement', 'nb_personnes_charge',
                   'secteur_activite']
    
    for row in rows:
        # Choisir une colonne au hasard pour ce row
        col = np.random.choice(missing_cols)
        df.at[row, col] = np.nan
    
    # 2. Valeurs aberrantes (environ 1% des données)
    outlier_pct = 0.01
    rows = np.random.choice(df.index, size=int(len(df) * outlier_pct), replace=False)
    
    for row in rows:
        # Choisir aléatoirement le type d'anomalie
        anomaly_type = np.random.choice(['age', 'revenu', 'endettement', 'experience'])
        
        if anomaly_type == 'age':
            # Âge impossible (très élevé)
            df.at[row, 'age'] = np.random.randint(100, 120)
        elif anomaly_type == 'revenu':
            # Revenu extrêmement élevé ou négatif
            if np.random.random() < 0.5:
                df.at[row, 'revenu_annuel'] = np.random.randint(1000000, 5000000)
            else:
                df.at[row, 'revenu_annuel'] = -np.random.randint(1000, 50000)
        elif anomaly_type == 'endettement':
            # Endettement impossible (> 100%)
            df.at[row, 'niveau_endettement'] = np.random.randint(100, 200)
        else:  # experience
            # Expérience > âge
            df.at[row, 'annees_experience'] = df.at[row, 'age'] + np.random.randint(1, 10)
    
    # 3. Colonnes redondantes
    # Créer une version discrétisée de l'âge
    df['age_groupe'] = pd.cut(df['age'], 
                             bins=[0, 25, 35, 45, 55, 65, 100],
                             labels=['18-25', '26-35', '36-45', '46-55', '56-65', '65+'])
    
    # Créer une version normalisée du revenu (redondante avec revenu_annuel)
    scaler = MinMaxScaler()
    df['revenu_normalise'] = scaler.fit_transform(df[['revenu_annuel']])
    
    # Créer une colonne agrégeant des informations existantes
    df['statut_socioeconomique'] = 'moyen'  # Valeur par défaut
    
    for idx, row in df.iterrows():
        score = 0
        
        # Points basés sur la catégorie socioprofessionnelle
        if row['categorie_sociopro'] in ['groupe_1', 'groupe_2']:
            score += 2
        elif row['categorie_sociopro'] == 'groupe_3':
            score += 1
        elif row['categorie_sociopro'] in ['groupe_5', 'groupe_6']:
            score -= 1
            
        # Points basés sur le niveau d'éducation
        if row['niveau_education'] == 'superieur':
            score += 2
        elif row['niveau_education'] == 'secondaire':
            score += 1
        elif row['niveau_education'] == 'sans_niveau':
            score -= 1
            
        # Points basés sur les possessions
        score += row['possession_voiture'] + row['possession_logement'] + \
                 row['possession_terrain'] + row['possession_investissements']
        
        # Déterminer le statut
        if score >= 4:
            df.at[idx, 'statut_socioeconomique'] = 'eleve'
        elif score >= 1:
            df.at[idx, 'statut_socioeconomique'] = 'moyen'
        else:
            df.at[idx, 'statut_socioeconomique'] = 'bas'
    
    # 4. Colonnes non pertinentes
    # Identifiant aléatoire
    df['id_personne'] = np.random.randint(10000, 99999, size=len(df))
    
    # Date de dernière mise à jour (non pertinente pour la prédiction)
    dates = []
    for _ in range(len(df)):
        days_ago = np.random.randint(0, 365)
        date = datetime.now() - timedelta(days=days_ago)
        dates.append(date.strftime('%Y-%m-%d'))
    df['date_maj'] = dates
    
    # Code postal aléatoire (non pertinent)
    df['code_postal'] = np.random.choice([10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000], size=len(df))

def main():
    """
    Fonction principale pour générer et sauvegarder le dataset.
    """
    # Générer le dataset
    df = generate_dataset()
    
    # Afficher quelques statistiques
    print("\nStatistiques du dataset généré:")
    print(f"Revenu moyen global: {df['revenu_annuel'].mean():.2f} DH/an")
    print(f"Revenu moyen urbain: {df[df['milieu'] == 'urbain']['revenu_annuel'].mean():.2f} DH/an")
    print(f"Revenu moyen rural: {df[df['milieu'] == 'rural']['revenu_annuel'].mean():.2f} DH/an")
    
    # Pourcentage en dessous de la moyenne
    mean_global = df['revenu_annuel'].mean()
    pct_below_mean = (df['revenu_annuel'] < mean_global).mean() * 100
    print(f"Pourcentage en dessous de la moyenne: {pct_below_mean:.1f}%")
    
    # Sauvegarder le dataset
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nDataset sauvegardé dans '{OUTPUT_FILE}'")
    
    return df

if __name__ == "__main__":
    main()