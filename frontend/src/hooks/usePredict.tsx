import {
  DefaultService,
  PatientInput,
  PatientInputDirect,
  type PredictionResponse,
  type PredictionResponseDirect,
} from "../service/api";

export default function usePredict() {
  const predictSentence = async (
    dto: PatientInput
  ): Promise<PredictionResponse> => {
    try {
      const response =
        await DefaultService.predictWithSentencePredictSentencePost(dto);
      return response;
    } catch (error) {
      console.error("Prediction error:", error);
      throw error;
    }
  };

  const predictDirect = async (
    dto: PatientInputDirect
  ): Promise<PredictionResponseDirect> => {
    try {
      const response = await DefaultService.predictDiagnosisPredictPost(dto);
      return response;
    } catch (error) {
      console.error("Prediction error:", error);
      throw error;
    }
  };

  return { predictSentence, predictDirect };
}
