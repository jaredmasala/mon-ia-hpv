"""
API Flask pour le dépistage CCU avec Primauté HPV - Microservice IA
Auteur: LUMANJI MBUNGA Luc & MASALA VAGULUA Jared
ISTA Kolwezi - Novembre 2025
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime
import joblib
import warnings
import json
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import skfuzzy as fuzz

warnings.filterwarnings('ignore')

# ============================================================================
# 1. SYSTÈME EXPERT AVEC PRIMAUTÉ VIROLOGIQUE HPV (Adapté pour API)
# ============================================================================

class HPVPrimacySystem:
    """
    Système expert avec primauté virologique du HPV
    Règle d'Or: Le CCU est causé à >99% par le HPV
    Deux régimes: A (HPV-) pour vulnérabilité, B (HPV+) pour dangerosité
    """
    
    # Configuration des pondérations hiérarchiques
    NIVEAU_1_POIDS = 0.70  # 70% pour les variables HPV
    NIVEAU_2_POIDS = 0.20  # 20% pour les aggravants
    NIVEAU_3_POIDS = 0.10  # 10% pour la vulnérabilité
    
    # Variables par niveau (poids relatifs)
    VARIABLES_NIVEAU_1 = {
        'test_hpv': 1.5,        # Présence du virus
        'genotype_hpv': 1.4,    # Type de HPV
        'charge_virale_hpv': 1.3 # Quantité virale
    }
    
    VARIABLES_NIVEAU_2 = {
        'test_vih': 1.1,          # Immunodépression majeure
        'nombre_mst': 1.2,        # Co-infections accumulées
        'antecedents_mst': 0.9,   # Histoire infectieuse
        'test_herpes': 0.8        # Co-infection virale
    }
    
    VARIABLES_NIVEAU_3 = {
        'nombre_partenaires': 1.0,
        'age': 0.4,
        'age_premier_rapport': 0.3,
        'contraceptifs_hormonaux': 0.1
    }
    
    # Classification des génotypes HPV par dangerosité
    GENOTYPE_CLASSES = {
        'CLASSE_1': [16, 18],      # Très haut risque oncogénique
        'CLASSE_2': [31, 33, 35, 45, 52, 58],  # Haut risque
        'CLASSE_3': [39, 51, 56, 59, 68],      # Risque intermédiaire
        'CLASSE_4': [26, 53, 66, 70, 73, 82]   # Faible risque
    }
    
    # Seuils de charge virale
    CHARGE_VIRALE_SEUILS = {
        'FAIBLE': (0, 100),
        'MODEREE': (100, 1000),
        'ELEVEE': (1000, 10000),
        'TRES_ELEVEE': (10000, 1000000)
    }
    
    # Classes de diagnostic final
    CLASSES_DIAGNOSTIC = {
        0: {'nom': 'Pas de risque', 'seuil_min': 0, 'seuil_max': 1.5, 'couleur': '#27ae60'},
        1: {'nom': 'Faible risque', 'seuil_min': 1.5, 'seuil_max': 3.0, 'couleur': '#2ecc71'},
        2: {'nom': 'Risque moyen', 'seuil_min': 3.0, 'seuil_max': 5.0, 'couleur': '#f39c12'},
        3: {'nom': 'Haut risque', 'seuil_min': 5.0, 'seuil_max': 7.0, 'couleur': '#e67e22'},
        4: {'nom': 'Très haut risque', 'seuil_min': 7.0, 'seuil_max': 10.0, 'couleur': '#c0392b'}
    }
    
    def analyser_patiente(self, patient_data: dict) -> dict:
        """Analyse principale avec logique à deux régimes"""
        # Étape 1: Vérifier primauté virologique
        test_hpv = patient_data.get('test_hpv', 0)
        
        if test_hpv == 0 or str(test_hpv).lower() in ['non', 'negatif', 'false']:
            return self._regime_a_vulnerabilite(patient_data)
        else:
            return self._regime_b_dangerosite(patient_data)
    
    def _regime_a_vulnerabilite(self, patient_data: dict) -> dict:
        """RÉGIME A: Patient HPV Négatif - VULNÉRABILITÉ"""
        # Calcul scores Niveaux 2 et 3
        score_n2 = self._calculer_score_niveau(patient_data, self.VARIABLES_NIVEAU_2)
        score_n3 = self._calculer_score_niveau(patient_data, self.VARIABLES_NIVEAU_3)
        
        score_total = (score_n2 * self.NIVEAU_2_POIDS + 
                      score_n3 * self.NIVEAU_3_POIDS)
        
        # Limitation à Risque Moyen maximum
        diagnostic = self._classifier_regime_a(score_total)
        
        return {
            'regime': 'A',
            'score_total': score_total,
            'score_n2': score_n2 * self.NIVEAU_2_POIDS,
            'score_n3': score_n3 * self.NIVEAU_3_POIDS,
            'score_n1': 0.0,
            'diagnostic': diagnostic,
            'interpretation': self._interpreter_regime_a(diagnostic, patient_data)
        }
    
    def _regime_b_dangerosite(self, patient_data: dict) -> dict:
        """RÉGIME B: Patient HPV Positif - DANGEROSITÉ"""
        # Évaluer dangerosité HPV
        dangerosite_hpv = self._evaluer_dangerosite_hpv(patient_data)
        
        # Calcul scores tous niveaux
        score_n1 = self._calculer_score_niveau(patient_data, self.VARIABLES_NIVEAU_1)
        score_n2 = self._calculer_score_niveau(patient_data, self.VARIABLES_NIVEAU_2)
        score_n3 = self._calculer_score_niveau(patient_data, self.VARIABLES_NIVEAU_3)
        
        # Score total avec dominance HPV
        score_total = (
            score_n1 * self.NIVEAU_1_POIDS * dangerosite_hpv['multiplicateur'] +
            score_n2 * self.NIVEAU_2_POIDS +
            score_n3 * self.NIVEAU_3_POIDS
        )
        
        # Transition vers Très Haut Risque si nécessaire
        score_total = self._appliquer_transition_risque(score_total, dangerosite_hpv, patient_data)
        
        diagnostic = self._classifier_regime_b(score_total)
        
        return {
            'regime': 'B',
            'score_total': score_total,
            'score_n1': score_n1 * self.NIVEAU_1_POIDS * dangerosite_hpv['multiplicateur'],
            'score_n2': score_n2 * self.NIVEAU_2_POIDS,
            'score_n3': score_n3 * self.NIVEAU_3_POIDS,
            'dangerosite_hpv': dangerosite_hpv,
            'diagnostic': diagnostic,
            'interpretation': self._interpreter_regime_b(diagnostic, dangerosite_hpv, patient_data)
        }
    
    def _calculer_score_niveau(self, patient_data: dict, variables: dict) -> float:
        """Calcul normalisé du score pour un niveau donné"""
        score = 0.0
        
        for variable, poids in variables.items():
            valeur = patient_data.get(variable, 0)
            valeur_normalisee = self._normaliser_variable(variable, valeur)
            score += poids * valeur_normalisee
        
        return min(score, 10.0)
    
    def _normaliser_variable(self, variable: str, valeur) -> float:
        """Normalisation spécifique selon le type de variable"""
        try:
            # Variables binaires
            if variable in ['test_hpv', 'test_vih', 'antecedents_mst', 'test_herpes', 'contraceptifs_hormonaux']:
                return 1.0 if str(valeur).lower() in ['oui', 'yes', '1', 'true', 'positif', 1] else 0.0
            
            # Charge virale HPV
            elif variable == 'charge_virale_hpv':
                vl = float(valeur)
                if vl == 0: return 0.0
                elif vl < 100: return 0.3
                elif vl < 1000: return 0.6
                elif vl < 10000: return 0.8
                else: return 1.0
            
            # Génotype HPV
            elif variable == 'genotype_hpv':
                genotype = int(float(valeur)) if valeur not in [None, '', 'NaN', '0'] else 0
                if genotype in self.GENOTYPE_CLASSES['CLASSE_1']: return 1.0
                elif genotype in self.GENOTYPE_CLASSES['CLASSE_2']: return 0.7
                elif genotype in self.GENOTYPE_CLASSES['CLASSE_3']: return 0.4
                else: return 0.2
            
            # Âge
            elif variable == 'age':
                age = float(valeur)
                if age < 25: return 0.8
                elif age <= 35: return 0.3
                elif age <= 45: return 0.6
                else: return 0.4
            
            # Nombre de partenaires
            elif variable == 'nombre_partenaires':
                nb = float(valeur)
                if nb <= 1: return 0.1
                elif nb <= 3: return 0.3
                elif nb <= 5: return 0.6
                else: return 1.0
            
            # Nombre de MST
            elif variable == 'nombre_mst':
                nb = float(valeur)
                return min(nb / 5.0, 1.0)
            
            # Âge premier rapport
            elif variable == 'age_premier_rapport':
                age = float(valeur)
                if age < 16: return 1.0
                elif age < 18: return 0.7
                elif age < 20: return 0.4
                else: return 0.1
            
            return float(valeur) if pd.notna(valeur) else 0.0
                
        except Exception as e:
            print(f"Erreur normalisation {variable}: {e}")
            return 0.0
    
    def _evaluer_dangerosite_hpv(self, patient_data: dict) -> dict:
        """Évaluer la dangerosité spécifique du HPV"""
        genotype = patient_data.get('genotype_hpv', 0)
        charge_virale = patient_data.get('charge_virale_hpv', 0)
        
        # Classification génotype
        classe_genotype = None
        for classe, genotypes in self.GENOTYPE_CLASSES.items():
            if genotype in genotypes:
                classe_genotype = classe
                break
        
        # Classification charge virale
        classe_charge = 'FAIBLE'
        try:
            cv = float(charge_virale)
            for nom_seuil, (min_val, max_val) in self.CHARGE_VIRALE_SEUILS.items():
                if min_val <= cv < max_val:
                    classe_charge = nom_seuil
                    break
        except:
            pass
        
        # Multiplicateur de dangerosité
        multiplicateur = 1.0
        if classe_genotype == 'CLASSE_1' and classe_charge in ['ELEVEE', 'TRES_ELEVEE']:
            multiplicateur = 1.8
        elif classe_genotype == 'CLASSE_1' and classe_charge == 'MODEREE':
            multiplicateur = 1.5
        
        return {
            'genotype': genotype,
            'classe_genotype': classe_genotype,
            'charge_virale': charge_virale,
            'classe_charge': classe_charge,
            'multiplicateur': multiplicateur
        }
    
    def _appliquer_transition_risque(self, score: float, dangerosite_hpv: dict, patient_data: dict) -> float:
        """Règles de transition vers 'Très Haut Risque'"""
        statut_vih = patient_data.get('test_vih', 0)
        antecedents_mst = patient_data.get('antecedents_mst', 0)
        nombre_mst = patient_data.get('nombre_mst', 0)
        
        # Condition principale
        condition_principale = (
            dangerosite_hpv['classe_genotype'] == 'CLASSE_1' and
            dangerosite_hpv['classe_charge'] in ['ELEVEE', 'TRES_ELEVEE']
        )
        
        # Facteurs aggravants
        facteurs_aggravants = []
        if statut_vih == 1:
            facteurs_aggravants.append('VIH+')
            score += 1.5
        if antecedents_mst == 1:
            facteurs_aggravants.append('Antécédents MST')
            score += 0.5
        if nombre_mst >= 3:
            facteurs_aggravants.append('Multiples MST')
            score += 0.8
        
        # Transition si conditions réunies
        if condition_principale and len(facteurs_aggravants) >= 2:
            score = max(score, 7.5)
        
        return min(score, 10.0)
    
    def _classifier_regime_a(self, score: float) -> dict:
        """Classification pour régime A (HPV négatif)"""
        score_limite = min(score, 5.0)
        
        for classe_id, info in self.CLASSES_DIAGNOSTIC.items():
            if info['seuil_min'] <= score_limite < info['seuil_max']:
                return {
                    'id': classe_id,
                    'nom': info['nom'],
                    'couleur': info['couleur'],
                    'score': score_limite,
                    'limite': True
                }
        
        return {
            'id': 2,
            'nom': 'Risque moyen',
            'couleur': '#f39c12',
            'score': score_limite,
            'limite': True
        }
    
    def _classifier_regime_b(self, score: float) -> dict:
        """Classification pour régime B (HPV positif)"""
        for classe_id, info in self.CLASSES_DIAGNOSTIC.items():
            if info['seuil_min'] <= score < info['seuil_max']:
                return {
                    'id': classe_id,
                    'nom': info['nom'],
                    'couleur': info['couleur'],
                    'score': score,
                    'limite': False
                }
        
        return {
            'id': 4,
            'nom': 'Très haut risque',
            'couleur': '#c0392b',
            'score': min(score, 10.0),
            'limite': False
        }
    
    def _interpreter_regime_a(self, diagnostic: dict, patient_data: dict) -> str:
        """Interprétation pour HPV négatif"""
        if diagnostic['id'] == 0:
            return "✓ Patient sans facteur de vulnérabilité significatif - Pas de risque immédiat"
        elif diagnostic['id'] == 1:
            return "⚠️ Vulnérabilité faible - Surveillance normale recommandée (dépistage tous les 3-5 ans)"
        else:
            return "⚠️ TERRAIN FRAGILE - Surveillance rapprochée nécessaire mais SANS danger immédiat de cancer"
    
    def _interpreter_regime_b(self, diagnostic: dict, dangerosite_hpv: dict, patient_data: dict) -> str:
        """Interprétation pour HPV positif"""
        classe_risque = diagnostic['id']
        
        interpretations = {
            0: "⚠️ INCOHÉRENCE: HPV positif mais classe 0 - Vérifier les données",
            1: "✓ HPV faible risque - Clairance virale probable sous surveillance",
            2: "⚠️ HPV avec facteurs de persistance - Surveillance rapprochée nécessaire",
            3: f"🔴 DANGER: HPV {dangerosite_hpv['classe_genotype']} + charge {dangerosite_hpv['classe_charge']} - Risque de lésions précancéreuses",
            4: "🔴 URGENCE: Transformation cellulaire probable - Risque élevé de cancer invasif"
        }
        
        return interpretations.get(classe_risque, "Indéterminé")
    
    def get_recommandations_cliniques(self, diagnostic: dict, regime: str, patient_data: dict) -> list:
        """Générer des recommandations cliniques adaptées"""
        recommandations = []
        
        if regime == 'A':  # HPV négatif
            if diagnostic['id'] == 0:
                recommandations.append("✓ Dépistage de routine selon directives nationales (tous les 3-5 ans)")
                recommandations.append("✓ Éducation sur la prévention des MST")
            elif diagnostic['id'] == 1:
                recommandations.append("⚠️ Dépistage renforcé (tous les 2-3 ans)")
                recommandations.append("⚠️ Évaluation des comportements à risque")
            else:  # Risque Moyen
                recommandations.append("🔴 Surveillance rapprochée (tous les 6-12 mois)")
                recommandations.append("🔴 Consultation gynécologique spécialisée")
                recommandations.append("🔴 Éducation intensive sur la prévention")
        
        else:  # HPV positif
            if diagnostic['id'] <= 1:
                recommandations.append("⚠️ Contrôle HPV dans 12 mois")
                recommandations.append("⚠️ Éviter les facteurs aggravants (tabac, autres MST)")
                recommandations.append("⚠️ Vaccination HPV si non vaccinée")
            elif diagnostic['id'] == 2:
                recommandations.append("🔴 Colposcopie recommandée dans les 6 mois")
                recommandations.append("🔴 Contrôle HPV semestriel")
                recommandations.append("🔴 Prise en charge des co-infections")
            elif diagnostic['id'] == 3:
                recommandations.append("🔴 URGENCE: Colposcopie + biopsie sous 1 mois")
                recommandations.append("🔴 Évaluation oncologique immédiate")
                recommandations.append("🔴 Prise en charge multidisciplinaire (gynéco-oncologue)")
            else:  # Classe 4: Très Haut Risque
                recommandations.append("🔴 URGENCE ABSOLUE: Consultation oncologique < 15 jours")
                recommandations.append("🔴 Biopsie systématique quelle que soit la colposcopie")
                recommandations.append("🔴 IRM pelvienne pour stadification")
        
        # Recommandations communes
        recommandations.append("📋 Ces recommandations doivent être adaptées au contexte clinique")
        recommandations.append("⚕️ Validation par un gynécologue-oncologue requise")
        
        return recommandations

# ============================================================================
# 2. MOTEUR IA AVEC PRIMAUTÉ HPV (Adapté pour API)
# ============================================================================

class FCMModelAPI:
    """Modèle FCM adapté pour API Web"""
    
    def __init__(self, model_path="fcm_model_primacy_auto_save.joblib"):
        self.systeme_expert = HPVPrimacySystem()
        self.model_loaded = False
        
        try:
            # Charger le modèle sauvegardé
            model_data = joblib.load(model_path)
            
            # Extraire les composants du modèle
            self.n_clusters = model_data.get('n_clusters', 5)
            self.m = model_data.get('m', 2.0)
            self.fcm_centers = model_data.get('fcm_centers')
            self.U = model_data.get('U')
            self.score_min = model_data.get('score_min', 0)
            self.score_max = model_data.get('score_max', 10)
            self.cluster_interpretation = model_data.get('cluster_interpretation', {})
            self.cluster_to_risk = model_data.get('cluster_to_risk', {})
            
            print(f"✅ Modèle chargé avec succès: {model_path}")
            print(f"   • Nombre de clusters: {self.n_clusters}")
            print(f"   • Score min/max: {self.score_min}/{self.score_max}")
            
            self.model_loaded = True
            
        except Exception as e:
            print(f"❌ Erreur chargement modèle: {str(e)}")
            self.model_loaded = False
    
    def prepare_features_for_fcm(self, patient_data, analyse_expert):
        """Préparer les features pour FCM basées sur l'analyse expert"""
        features = []
        
        # Score expert normalisé
        features.append(analyse_expert['score_total'] / 10.0)
        
        # Régime (0 = A, 1 = B)
        features.append(1.0 if analyse_expert['regime'] == 'B' else 0.0)
        
        # Distribution des scores par niveau
        features.append(analyse_expert.get('score_n1', 0) / 10.0)
        features.append(analyse_expert.get('score_n2', 0) / 10.0)
        features.append(analyse_expert.get('score_n3', 0) / 10.0)
        
        # Variables HPV importantes
        features.append(1.0 if patient_data.get('test_hpv', 0) == 1 else 0.0)
        
        # Génotype dangerosité
        genotype = patient_data.get('genotype_hpv', 0)
        if genotype in self.systeme_expert.GENOTYPE_CLASSES['CLASSE_1']:
            genotype_score = 1.0
        elif genotype in self.systeme_expert.GENOTYPE_CLASSES['CLASSE_2']:
            genotype_score = 0.7
        else:
            genotype_score = 0.3
        features.append(genotype_score)
        
        # Charge virale normalisée
        charge_virale = patient_data.get('charge_virale_hpv', 0)
        try:
            cv_norm = np.log10(float(charge_virale) + 1) / 6.0
        except:
            cv_norm = 0.0
        features.append(cv_norm)
        
        # Statut VIH
        features.append(1.0 if patient_data.get('test_vih', 0) == 1 else 0.0)
        
        return np.array(features).reshape(1, -1)
    
    def calculate_fuzzy_membership(self, score_norm):
        """Calcule les appartenances floues pour un score normalisé"""
        if self.fcm_centers is None:
            return np.zeros(self.n_clusters)
        
        # Calcul des distances aux centres (1D)
        distances = np.abs(self.fcm_centers.flatten() - score_norm)
        
        # Appartenance floue (inverse de la distance)
        distances = np.maximum(distances, 1e-10)
        memberships = 1 / distances
        
        # Normaliser pour somme = 1
        memberships = memberships / np.sum(memberships)
        
        return memberships.flatten()
    
    def predict(self, patient_data):
        """
        Prédiction pour un patient
        Retourne un dictionnaire avec les résultats complets
        """
        if not self.model_loaded:
            return {'error': 'Modèle non chargé'}
        
        try:
            # 1. Analyse système expert avec primauté HPV
            analyse_expert = self.systeme_expert.analyser_patiente(patient_data)
            
            # 2. Score système expert
            expert_score = analyse_expert['score_total']
            
            # 3. Normalisation pour FCM
            score_norm = (expert_score - self.score_min) / (self.score_max - self.score_min + 1e-10)
            
            # 4. Calcul des appartenances floues
            fuzzy_memberships = self.calculate_fuzzy_membership(score_norm)
            
            # 5. Décision floue
            cluster_decision = np.argmax(fuzzy_memberships)
            confidence = fuzzy_memberships[cluster_decision]
            
            # 6. Récupérer l'interprétation du cluster
            cluster_info = self.cluster_interpretation.get(int(cluster_decision), {})
            
            # 7. Fusion : 60% expert, 40% FCM
            expert_weight = 0.6
            fcm_weight = 0.4
            
            # Convertir cluster en score (0-10)
            cluster_score = cluster_info.get('niveau_risque', 2) * 2.5
            
            final_score = (expert_weight * expert_score + fcm_weight * cluster_score)
            final_score = max(0, min(final_score, 10))
            
            # 8. Classification finale
            classe_finale = self.classifier_final(final_score, analyse_expert)
            
            # 9. Recommandations adaptées
            recommandations = self.systeme_expert.get_recommandations_cliniques(
                classe_finale, analyse_expert['regime'], patient_data
            )
            
            # 10. Niveau risque HPV
            genotype = patient_data.get('genotype_hpv', 0)
            hpv_risk_level = self.get_hpv_risk_level(genotype)
            
            # 11. Charge virale
            hpv_viral_load_level = self.get_viral_load_level(
                patient_data.get('charge_virale_hpv', 0)
            )
            
            # Résultats complets
            results = {
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
                
                # Scores et classification
                'expert_score': float(expert_score),
                'final_score': float(final_score),
                'risk_level': classe_finale['nom'],
                'risk_color': classe_finale['couleur'],
                'regime': analyse_expert['regime'],
                
                # Informations FCM
                'fcm': {
                    'cluster_id': int(cluster_decision),
                    'cluster_name': cluster_info.get('nom', 'Inconnu'),
                    'membership_confidence': float(confidence),
                    'membership_distribution': fuzzy_memberships.tolist()
                },
                
                # Informations HPV
                'hpv': {
                    'hpv_status': 'positif' if patient_data.get('test_hpv', 0) == 1 else 'negatif',
                    'genotype': int(genotype),
                    'genotype_risk': hpv_risk_level,
                    'viral_load': patient_data.get('charge_virale_hpv', 0),
                    'viral_load_level': hpv_viral_load_level
                },
                
                # Recommandations
                'recommendations': recommandations,
                
                # Analyse détaillée
                'analysis': {
                    'interpretation': analyse_expert['interpretation'],
                    'scores_breakdown': {
                        'niveau_1_hpv': float(analyse_expert.get('score_n1', 0)),
                        'niveau_2_aggravants': float(analyse_expert.get('score_n2', 0)),
                        'niveau_3_vulnerabilite': float(analyse_expert.get('score_n3', 0))
                    }
                }
            }
            
            return results
            
        except Exception as e:
            print(f"❌ Erreur prédiction: {str(e)}")
            print(traceback.format_exc())
            return {'error': f'Erreur prédiction: {str(e)}'}
    
    def classifier_final(self, score, analyse_expert):
        """Classification finale avec vérification de cohérence"""
        # Si régime A (HPV négatif), limiter à Risque Moyen
        if analyse_expert['regime'] == 'A':
            score = min(score, 5.0)
        
        for classe_id, info in self.systeme_expert.CLASSES_DIAGNOSTIC.items():
            if info['seuil_min'] <= score < info['seuil_max']:
                return {
                    'id': classe_id,
                    'nom': info['nom'],
                    'couleur': info['couleur'],
                    'score': score,
                    'regime': analyse_expert['regime']
                }
        
        # Par défaut
        return {
            'id': 2,
            'nom': 'Risque moyen',
            'couleur': '#f39c12',
            'score': score,
            'regime': analyse_expert['regime']
        }
    
    def get_hpv_risk_level(self, genotype):
        """Déterminer le niveau de risque d'un génotype HPV"""
        try:
            gen_val = int(float(genotype)) if genotype not in [None, '', 'NaN', '0'] else 0
            if gen_val in self.systeme_expert.GENOTYPE_CLASSES['CLASSE_1']:
                return 'très_haut_risque'
            elif gen_val in self.systeme_expert.GENOTYPE_CLASSES['CLASSE_2']:
                return 'haut_risque'
            elif gen_val in self.systeme_expert.GENOTYPE_CLASSES['CLASSE_3']:
                return 'risque_intermédiaire'
            else:
                return 'faible_risque'
        except:
            return 'faible_risque'
    
    def get_viral_load_level(self, viral_load):
        """Catégoriser la charge virale HPV"""
        try:
            vl_val = float(viral_load)
            for level, (min_val, max_val) in self.systeme_expert.CHARGE_VIRALE_SEUILS.items():
                if min_val <= vl_val < max_val:
                    return level
            return 'FAIBLE'
        except:
            return 'FAIBLE'

# ============================================================================
# 3. APPLICATION FLASK
# ============================================================================

app = Flask(__name__)
CORS(app)  # Autoriser les requêtes cross-origin

# Initialiser le modèle
print("🔄 Initialisation du modèle FCM avec primauté HPV...")
model = FCMModelAPI("fcm_model_primacy_auto_save.joblib")

@app.route('/')
def home():
    """Page d'accueil de l'API"""
    return jsonify({
        'status': 'online',
        'service': 'API Dépistage CCU - Primauté HPV',
        'version': '1.0.0',
        'author': 'ISTA Kolwezi',
        'model_loaded': model.model_loaded
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """Endpoint de vérification de santé"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_status': 'loaded' if model.model_loaded else 'not_loaded'
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Endpoint de prédiction
    Attend un JSON avec les données patient
    """
    try:
        # Récupérer les données
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'Aucune donnée reçue'}), 400
        
        # Valider les données requises
        required_fields = ['age', 'test_hpv']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Champ requis manquant: {field}'}), 400
        
        # Préparer les données patient
        patient_data = {
            # Données démographiques (Niveau 3)
            'age': float(data.get('age', 30)),
            'nombre_partenaires': float(data.get('nombre_partenaires', 0)),
            'age_premier_rapport': float(data.get('age_premier_rapport', 18)),
            'contraceptifs_hormonaux': 1 if str(data.get('contraceptifs_hormonaux', 'non')).lower() in ['oui', 'yes', '1', 'true'] else 0,
            
            # Antécédents gynécologiques
            'nombre_grossesses': float(data.get('nombre_grossesses', 0)),
            'nombre_ivg': float(data.get('nombre_ivg', 0)),
            
            # Infections (Niveau 2)
            'test_vih': 1 if str(data.get('test_vih', 'non')).lower() in ['oui', 'yes', '1', 'true', 'positif'] else 0,
            'antecedents_mst': 1 if str(data.get('antecedents_mst', 'non')).lower() in ['oui', 'yes', '1', 'true'] else 0,
            'nombre_mst': float(data.get('nombre_mst', 0)),
            'test_herpes': 1 if str(data.get('test_herpes', 'non')).lower() in ['oui', 'yes', '1', 'true'] else 0,
            
            # HPV - Niveau 1 (Primauté)
            'test_hpv': 1 if str(data.get('test_hpv', 'non')).lower() in ['oui', 'yes', '1', 'true', 'positif'] else 0,
            'genotype_hpv': int(float(data.get('genotype_hpv', 0))),
            'charge_virale_hpv': float(data.get('charge_virale_hpv', 0))
        }
        
        # Faire la prédiction
        result = model.predict(patient_data)
        
        # Ajouter les données d'entrée pour référence
        result['input_data'] = patient_data
        
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ Erreur endpoint /predict: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'error': f'Erreur interne: {str(e)}',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/clusters', methods=['GET'])
def get_clusters_info():
    """Endpoint pour obtenir les informations des clusters"""
    if not model.model_loaded:
        return jsonify({'error': 'Modèle non chargé'}), 400
    
    clusters_info = []
    for cluster_id, info in model.cluster_interpretation.items():
        clusters_info.append({
            'id': cluster_id,
            'name': info.get('nom', 'Inconnu'),
            'color': info.get('couleur', '#cccccc'),
            'risk_level': info.get('niveau_risque', 2),
            'stats': info.get('stats', {})
        })
    
    return jsonify({
        'status': 'success',
        'n_clusters': model.n_clusters,
        'clusters': clusters_info
    })

@app.route('/api/model-info', methods=['GET'])
def get_model_info():
    """Endpoint pour obtenir les informations du modèle"""
    return jsonify({
        'status': 'success',
        'model_loaded': model.model_loaded,
        'n_clusters': model.n_clusters,
        'm_parameter': model.m,
        'score_range': {'min': float(model.score_min), 'max': float(model.score_max)},
        'primauté_hpv': True,
        'architecture': 'Système Expert + Fuzzy C-Means'
    })

if __name__ == '__main__':
    # Configuration
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
    
    print(f"🚀 Démarrage de l'API Dépistage CCU sur le port {port}")
    print(f"📊 Modèle chargé: {'✅' if model.model_loaded else '❌'}")
    print(f"🔗 Endpoints disponibles:")
    print(f"   • GET  /              - Page d'accueil")
    print(f"   • GET  /api/health    - Vérification santé")
    print(f"   • POST /api/predict   - Prédiction (données JSON)")
    print(f"   • GET  /api/clusters  - Informations clusters")
    print(f"   • GET  /api/model-info - Informations modèle")
    
    app.run(host='0.0.0.0', port=port, debug=debug)
    
