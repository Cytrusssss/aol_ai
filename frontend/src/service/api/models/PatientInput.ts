/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
/**
 * Patient input with text-based symptom description
 */
export type PatientInput = {
    /**
     * Patient age
     */
    Age: number;
    Gender: PatientInput.Gender;
    /**
     * Describe your symptoms in natural language
     */
    symptoms_description: string;
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
export namespace PatientInput {
    export enum Gender {
        MALE = 'Male',
        FEMALE = 'Female',
    }
}

