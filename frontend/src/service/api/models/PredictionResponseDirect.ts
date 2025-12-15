/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 * Prediction response for direct symptom input (no extraction)
 */
export type PredictionResponseDirect = {
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
};

