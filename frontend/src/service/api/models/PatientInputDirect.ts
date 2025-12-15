/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 * Patient input with directly selected symptoms (not natural language)
 */
export type PatientInputDirect = {
    /**
     * Patient age
     */
    Age: number;
    Gender: PatientInputDirect.Gender;
    Symptom_1: PatientInputDirect.Symptom_1;
    Symptom_2: PatientInputDirect.Symptom_2;
    Symptom_3: PatientInputDirect.Symptom_3;
    /**
     * Heart rate in BPM
     */
    Heart_Rate_bpm: number;
    /**
     * Body temperature in Celsius
     */
    Body_Temperature_C: number;
    /**
     * Oxygen saturation percentage
     */
    'Oxygen_Saturation_%': number;
    /**
     * Systolic blood pressure
     */
    Systolic: number;
    /**
     * Diastolic blood pressure
     */
    Diastolic: number;
};
export namespace PatientInputDirect {
    export enum Gender {
        MALE = 'Male',
        FEMALE = 'Female',
    }
    export enum Symptom_1 {
        FATIGUE = 'Fatigue',
        SORE_THROAT = 'Sore throat',
        BODY_ACHE = 'Body ache',
        SHORTNESS_OF_BREATH = 'Shortness of breath',
        RUNNY_NOSE = 'Runny nose',
        HEADACHE = 'Headache',
        COUGH = 'Cough',
        FEVER = 'Fever',
    }
    export enum Symptom_2 {
        FATIGUE = 'Fatigue',
        SORE_THROAT = 'Sore throat',
        BODY_ACHE = 'Body ache',
        SHORTNESS_OF_BREATH = 'Shortness of breath',
        RUNNY_NOSE = 'Runny nose',
        HEADACHE = 'Headache',
        COUGH = 'Cough',
        FEVER = 'Fever',
    }
    export enum Symptom_3 {
        FATIGUE = 'Fatigue',
        SORE_THROAT = 'Sore throat',
        BODY_ACHE = 'Body ache',
        SHORTNESS_OF_BREATH = 'Shortness of breath',
        RUNNY_NOSE = 'Runny nose',
        HEADACHE = 'Headache',
        COUGH = 'Cough',
        FEVER = 'Fever',
    }
}

