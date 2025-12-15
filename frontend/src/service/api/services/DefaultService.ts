/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { PatientInput } from '../models/PatientInput';
import type { PatientInputDirect } from '../models/PatientInputDirect';
import type { PredictionResponse } from '../models/PredictionResponse';
import type { PredictionResponseDirect } from '../models/PredictionResponseDirect';
import type { CancelablePromise } from '../core/CancelablePromise';
import { OpenAPI } from '../core/OpenAPI';
import { request as __request } from '../core/request';
export class DefaultService {
    /**
     * Root
     * @returns any Successful Response
     * @throws ApiError
     */
    public static rootGet(): CancelablePromise<any> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/',
        });
    }
    /**
     * Health
     * @returns any Successful Response
     * @throws ApiError
     */
    public static healthHealthGet(): CancelablePromise<any> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/health',
        });
    }
    /**
     * Predict With Sentence
     * Predict diagnosis based on natural language symptom description and vital signs.
     *
     * Simply describe your symptoms naturally (e.g., "I have a cough, fever and headache"),
     * and the system will:
     * 1. Extract the top 3 matching symptoms using Jaro-Winkler similarity
     * 2. Analyze your vital signs
     * 3. Predict the most likely diagnosis
     *
     * Example input:
     * ```json
     * {
         * "Age": 45,
         * "Gender": "Male",
         * "symptoms_description": "I have a bad cough, high fever, and I feel very tired",
         * "Heart_Rate_bpm":  85,
         * "Body_Temperature_C": 38.5,
         * "Oxygen_Saturation_%": 96,
         * "Systolic": 120,
         * "Diastolic": 80
         * }
         * ```
         * @param requestBody
         * @returns PredictionResponse Successful Response
         * @throws ApiError
         */
        public static predictWithSentencePredictSentencePost(
            requestBody: PatientInput,
        ): CancelablePromise<PredictionResponse> {
            return __request(OpenAPI, {
                method: 'POST',
                url: '/predict/sentence',
                body: requestBody,
                mediaType: 'application/json',
                errors: {
                    422: `Validation Error`,
                },
            });
        }
        /**
         * Predict Diagnosis
         * Predict diagnosis with directly selected symptoms (not natural language).
         *
         * Use this endpoint when you already know the exact symptoms to select.
         * For natural language input, use /predict/sentence instead.
         *
         * Example input:
         * ```json
         * {
             * "Age": 45,
             * "Gender": "Male",
             * "Symptom_1": "Fever",
             * "Symptom_2": "Cough",
             * "Symptom_3":  "Fatigue",
             * "Heart_Rate_bpm": 85,
             * "Body_Temperature_C": 38.5,
             * "Oxygen_Saturation_%": 96,
             * "Systolic": 120,
             * "Diastolic":  80
             * }
             * ```
             *
             * Returns:
             * - predicted_diagnosis: The most likely diagnosis
             * - confidence: Probability of the predicted diagnosis
             * - all_probabilities:  Probabilities for all diagnoses
             * @param requestBody
             * @returns PredictionResponseDirect Successful Response
             * @throws ApiError
             */
            public static predictDiagnosisPredictPost(
                requestBody: PatientInputDirect,
            ): CancelablePromise<PredictionResponseDirect> {
                return __request(OpenAPI, {
                    method: 'POST',
                    url: '/predict',
                    body: requestBody,
                    mediaType: 'application/json',
                    errors: {
                        422: `Validation Error`,
                    },
                });
            }
        }
