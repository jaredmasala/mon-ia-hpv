<?php
/**
 * Script PHP de liaison pour le système de dépistage CCU avec Primauté HPV
 * Auteur: ISTA Kolwezi
 * Date: Novembre 2025
 */

header('Content-Type: application/json');
header('Access-Control-Allow-Origin: *');
header('Access-Control-Allow-Methods: POST, OPTIONS');
header('Access-Control-Allow-Headers: Content-Type');

// Gérer les requêtes OPTIONS pour CORS
if ($_SERVER['REQUEST_METHOD'] === 'OPTIONS') {
    http_response_code(200);
    exit();
}

// Configuration
$API_URL = 'https://ia-depistage-hpv1.onrender.com'; // URL de l'API Flask
$MAX_AGE = 80;
$MIN_AGE = 15;

// Fonction de validation et nettoyage
function validate_input($data) {
    $errors = [];
    
    // Validation de l'âge
    if (!isset($data['age']) || !is_numeric($data['age'])) {
        $errors[] = "L'âge est requis et doit être numérique";
    } else {
        $age = (int)$data['age'];
        if ($age < 15 || $age > 80) {
            $errors[] = "L'âge doit être entre 15 et 80 ans";
        }
    }
    
    // Validation HPV
    if (!isset($data['test_hpv'])) {
        $errors[] = "Le statut HPV est requis";
    }
    
    // Validation des champs numériques
    $numeric_fields = [
        'nombre_partenaires', 'age_premier_rapport', 'nombre_grossesses',
        'nombre_ivg', 'nombre_mst', 'genotype_hpv', 'charge_virale_hpv'
    ];
    
    foreach ($numeric_fields as $field) {
        if (isset($data[$field]) && !is_numeric($data[$field])) {
            $errors[] = "Le champ $field doit être numérique";
        }
    }
    
    return $errors;
}

// Fonction de nettoyage
function sanitize_input($data) {
    $sanitized = [];
    
    foreach ($data as $key => $value) {
        if (is_string($value)) {
            // Nettoyage basique
            $sanitized[$key] = htmlspecialchars(strip_tags(trim($value)), ENT_QUOTES, 'UTF-8');
        } else {
            $sanitized[$key] = $value;
        }
    }
    
    return $sanitized;
}

// Fonction pour envoyer la requête à l'API Python
function call_python_api($data, $api_url) {
    $ch = curl_init();
    
    curl_setopt($ch, CURLOPT_URL, $api_url);
    curl_setopt($ch, CURLOPT_POST, 1);
    curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($data));
    curl_setopt($ch, CURLOPT_HTTPHEADER, ['Content-Type: application/json']);
    curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
    curl_setopt($ch, CURLOPT_TIMEOUT, 30); // Timeout de 30 secondes
    curl_setopt($ch, CURLOPT_CONNECTTIMEOUT, 10); // Timeout de connexion
    
    $response = curl_exec($ch);
    $http_code = curl_getinfo($ch, CURLINFO_HTTP_CODE);
    $error = curl_error($ch);
    
    curl_close($ch);
    
    if ($error) {
        return [
            'success' => false,
            'error' => "Erreur CURL: " . $error
        ];
    }
    
    return [
        'success' => true,
        'http_code' => $http_code,
        'response' => json_decode($response, true)
    ];
}

// Point d'entrée principal
try {
    // Vérifier la méthode
    if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
        http_response_code(405);
        echo json_encode([
            'success' => false,
            'error' => 'Méthode non autorisée. Utilisez POST.'
        ]);
        exit();
    }
    
    // Récupérer les données JSON
    $json_input = file_get_contents('php://input');
    $input_data = json_decode($json_input, true);
    
    if (json_last_error() !== JSON_ERROR_NONE) {
        http_response_code(400);
        echo json_encode([
            'success' => false,
            'error' => 'JSON invalide: ' . json_last_error_msg()
        ]);
        exit();
    }
    
    // Nettoyer les données
    $clean_data = sanitize_input($input_data);
    
    // Valider les données
    $validation_errors = validate_input($clean_data);
    
    if (!empty($validation_errors)) {
        http_response_code(400);
        echo json_encode([
            'success' => false,
            'error' => 'Erreurs de validation',
            'details' => $validation_errors
        ]);
        exit();
    }
    
    // Préparer les données pour l'API
    $api_data = [
        'age' => (float)$clean_data['age'],
        'test_hpv' => $clean_data['test_hpv'],
        
        // Données démographiques
        'nombre_partenaires' => isset($clean_data['nombre_partenaires']) ? (float)$clean_data['nombre_partenaires'] : 0,
        'age_premier_rapport' => isset($clean_data['age_premier_rapport']) ? (float)$clean_data['age_premier_rapport'] : 18,
        'contraceptifs_hormonaux' => isset($clean_data['contraceptifs_hormonaux']) ? $clean_data['contraceptifs_hormonaux'] : 'non',
        
        // Antécédents gynécologiques
        'nombre_grossesses' => isset($clean_data['nombre_grossesses']) ? (float)$clean_data['nombre_grossesses'] : 0,
        'nombre_ivg' => isset($clean_data['nombre_ivg']) ? (float)$clean_data['nombre_ivg'] : 0,
        
        // Infections
        'test_vih' => isset($clean_data['test_vih']) ? $clean_data['test_vih'] : 'non',
        'antecedents_mst' => isset($clean_data['antecedents_mst']) ? $clean_data['antecedents_mst'] : 'non',
        'nombre_mst' => isset($clean_data['nombre_mst']) ? (float)$clean_data['nombre_mst'] : 0,
        'test_herpes' => isset($clean_data['test_herpes']) ? $clean_data['test_herpes'] : 'non',
        
        // HPV
        'genotype_hpv' => isset($clean_data['genotype_hpv']) ? (float)$clean_data['genotype_hpv'] : 0,
        'charge_virale_hpv' => isset($clean_data['charge_virale_hpv']) ? (float)$clean_data['charge_virale_hpv'] : 0,
        
        // Métadonnées
        'timestamp' => date('Y-m-d H:i:s'),
        'source' => 'web_app'
    ];
    
    // Appeler l'API Python
    $api_result = call_python_api($api_data, $API_URL);
    
    if (!$api_result['success']) {
        http_response_code(502);
        echo json_encode([
            'success' => false,
            'error' => 'Erreur de communication avec le serveur IA',
            'details' => $api_result['error']
        ]);
        exit();
    }
    
    // Vérifier le code HTTP
    if ($api_result['http_code'] !== 200) {
        http_response_code(502);
        echo json_encode([
            'success' => false,
            'error' => "L'API IA a retourné une erreur",
            'http_code' => $api_result['http_code'],
            'api_response' => $api_result['response']
        ]);
        exit();
    }
    
    // Retourner la réponse
    http_response_code(200);
    echo json_encode([
        'success' => true,
        'timestamp' => date('Y-m-d H:i:s'),
        'data' => $api_result['response']
    ]);
    
} catch (Exception $e) {
    // Gestion des erreurs générales
    http_response_code(500);
    echo json_encode([
        'success' => false,
        'error' => 'Erreur interne du serveur',
        'message' => $e->getMessage()
    ]);
}
?>
