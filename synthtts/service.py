import TTS
import wave
import contextlib
import base64
from TTS.utils.manage import ModelManager
from pathlib import Path
from .synth import Synthesizer


def synthesize_tts(
    text,
    output_path,
    model_name="tts_models/en/ljspeech/glow-tts",
    vocoder_name=None,
    model_path=None,
    config_path=None,
    vocoder_path=None,
    vocoder_config_path=None,
    encoder_path=None,
    encoder_config_path=None,
    speakers_file_path=None,
    language_ids_file_path=None,
    speaker_idx=None,
    language_idx=None,
    speaker_wav=None,
    capacitron_style_wav=None,
    capacitron_style_text=None,
    reference_wav=None,
    reference_speaker_idx=None,
    use_cuda=False,
):
    """
    Function to synthesize speech using Text-to-Speech (TTS) models.

    Parameters:
    - text (str): The text to be synthesized.
    - output_path (str): The path where the synthesized audio will be stored.
    - model_name (str, optional): Name of the TTS model to be used.
    - ......

    Returns:
    - str: Path to the synthesized audio file.
    - list: List containing word boundaries for the synthesized speech.
    """

    # load model managera
    print("looking for model manager")
    print(  Path( TTS.__file__).parent )

    path = Path(TTS.__file__).parent / ".models.json"

    manager = ModelManager(path)

    # CASE3: load pre-trained model paths
    if model_name is not None and not model_path:
        model_path, config_path, model_item = manager.download_model(model_name)
        vocoder_name = (
            model_item["default_vocoder"] if vocoder_name is None else vocoder_name
        )

    if vocoder_name is not None and not vocoder_path:
        vocoder_path, vocoder_config_path, _ = manager.download_model(vocoder_name)

    # load models
    synthesizer = Synthesizer(
        model_path,
        config_path,
        speakers_file_path,
        language_ids_file_path,
        vocoder_path,
        vocoder_config_path,
        encoder_path,
        encoder_config_path,
        use_cuda,
    )

    # RUN THE SYNTHESIS
    print(" > Text: {}".format(text))
    wav, word_boundaries = synthesizer.tts(
        text,
        speaker_idx,
        language_idx,
        speaker_wav,
        reference_wav=reference_wav,
        style_wav=capacitron_style_wav,
        style_text=capacitron_style_text,
        reference_speaker_name=reference_speaker_idx,
    )
    # save the results
    print(" > Saving output to {}".format(output_path))

    # Replace file extension with .wav
    wav_path = Path(output_path).with_suffix(".wav")
    synthesizer.save_wav(wav, wav_path)
    # wav2mp3(wav_path, output_path)

    return wav_path, word_boundaries


class TtsService():
    """
    Class for Text-to-Speech (TTS) services.
    """

    def __init__(
        self,
        **kwargs,
    ):
        """
        Initializes the TTS Service with the provided parameters.
        """
        self.init_kwargs = kwargs

    def generate_from_text(
        self, text: str, cache_dir: str = None, fname: str = "output.wav", **kwargs
    ) -> dict:
        """
        Function to generate speech from text.

        Parameters:
        - text (str): The text to be synthesized
        - cache_dir (str, optional): The directory for caching audio files.
        - fname (str, optional): The filename for the generated audio file.

        Returns:
        - dict: Dictionary with the details of the synthesized speech, including word boundaries and audio file details.
        """

        if cache_dir is None:
            cache_dir = self.cache_dir

        input_text = text
        input_data = {"input_text": text, "service": "coqui"}

        if not kwargs:
            kwargs = self.init_kwargs

        _, word_boundaries = synthesize_tts(
            input_text, str(Path(cache_dir) / fname), **kwargs
        )

        json_dict = {
            "input_text": text,
            "input_data": input_data,
            "original_audio": fname,
            "word_boundaries": word_boundaries,
        }

        try:
            with open(fname, "rb") as f:
                binary = f.read()
                json_dict["base64"] = base64.b64encode(binary).decode('utf-8')
        except FileNotFoundError:
            json_dict["error"] = f"[!] WAV file not found: {fname}"

        with contextlib.closing(wave.open(fname, 'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            json_dict["frames"] = frames
            json_dict["rate"] = rate
            json_dict["duration"] = frames / rate

        return json_dict
