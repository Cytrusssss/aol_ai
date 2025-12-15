/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 * Prediction response with diagnosis and extracted symptoms
 */
export type PredictionResponse = {
    /**
     * The predicted diagnosis
     */
    predicted_diagnosis: string;
    /**
     * Index of predicted class
     */
    predicted_class_index: number;
    /**
     * Confidence score (0-1)
     */
    confidence: number;
    /**
     * Probabilities for all diagnoses
     */
    all_probabilities: Record<string, number>;
    /**
     * Symptoms extracted from description
     */
    extracted_symptoms: Array<string>;
    /**
     * Similarity scores for extracted symptoms
     */
    extraction_scores: Record<string, number>;
};

