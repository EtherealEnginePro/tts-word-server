import os
import json
import hashlib

import wave
import contextlib
fname = './output.wav'

from pprint import pprint
from pathlib import Path

import time
from typing import List
import numpy as np
import pysbd
import torch

# from pydub import AudioSegment

# from manim_voiceover.helper import prompt_ask_missing_package, remove_bookmarks
# from manim_voiceover.services.base import SpeechService
# from manim_voiceover.services.coqui.synthesize import synthesize_tts, DEFAULT_MODEL
# from manim_voiceover.helper import wav2mp3

# DEFAULT_MODEL = "tts_models/en/ljspeech/tacotron2-DDC"
DEFAULT_MODEL = "tts_models/en/ljspeech/glow-tts"
AUDIO_OFFSET_RESOLUTION = 1000

try:
    import TTS
    from TTS.utils.manage import ModelManager
    # from .utils_synthesizer import Synthesizer
except ImportError:
    print("Missing packages. Run `pip install TTS` to use TtsService.")

from TTS.config import load_config
from TTS.tts.models import setup_model as setup_tts_model

# pylint: disable=unused-wildcard-import
# pylint: disable=wildcard-import
# from .tts_utils_synthesis import synthesis, transfer_voice, trim_silence
from TTS.tts.utils.synthesis import synthesis, transfer_voice, trim_silence
from TTS.utils.audio import AudioProcessor
from TTS.vocoder.models import setup_model as setup_vocoder_model
from TTS.vocoder.utils.generic_utils import interpolate_vocoder_input

# from manim_voiceover.tracker import AUDIO_OFFSET_RESOLUTION

# Forking the same class from Coqui for now
class Synthesizer(object):
    def __init__(
        self,
        tts_checkpoint: str,
        tts_config_path: str,
        tts_speakers_file: str = "",
        tts_languages_file: str = "",
        vocoder_checkpoint: str = "",
        vocoder_config: str = "",
        encoder_checkpoint: str = "",
        encoder_config: str = "",
        use_cuda: bool = False,
    ) -> None:
        """General 🐸 TTS interface for inference. It takes a tts and a vocoder
        model and synthesize speech from the provided text.

        The text is divided into a list of sentences using `pysbd` and synthesize
        speech on each sentence separately.

        If you have certain special characters in your text, you need to handle
        them before providing the text to Synthesizer.

        TODO: set the segmenter based on the source language

        Args:
            tts_checkpoint (str): path to the tts model file.
            tts_config_path (str): path to the tts config file.
            vocoder_checkpoint (str, optional): path to the vocoder model file. Defaults to None.
            vocoder_config (str, optional): path to the vocoder config file. Defaults to None.
            encoder_checkpoint (str, optional): path to the speaker encoder model file. Defaults to `""`,
            encoder_config (str, optional): path to the speaker encoder config file. Defaults to `""`,
            use_cuda (bool, optional): enable/disable cuda. Defaults to False.
        """
        self.tts_checkpoint = tts_checkpoint
        self.tts_config_path = tts_config_path
        self.tts_speakers_file = tts_speakers_file
        self.tts_languages_file = tts_languages_file
        self.vocoder_checkpoint = vocoder_checkpoint
        self.vocoder_config = vocoder_config
        self.encoder_checkpoint = encoder_checkpoint
        self.encoder_config = encoder_config
        self.use_cuda = use_cuda

        self.tts_model = None
        self.vocoder_model = None
        self.speaker_manager = None
        self.num_speakers = 0
        self.tts_speakers = {}
        self.language_manager = None
        self.num_languages = 0
        self.tts_languages = {}
        self.d_vector_dim = 0
        self.seg = self._get_segmenter("en")
        self.use_cuda = use_cuda

        if self.use_cuda:
            assert torch.cuda.is_available(), "CUDA is not availabe on this machine."
        self._load_tts(tts_checkpoint, tts_config_path, use_cuda)
        self.output_sample_rate = self.tts_config.audio["sample_rate"]
        if vocoder_checkpoint:
            self._load_vocoder(vocoder_checkpoint, vocoder_config, use_cuda)
            self.output_sample_rate = self.vocoder_config.audio["sample_rate"]

    @staticmethod
    def _get_segmenter(lang: str):
        """get the sentence segmenter for the given language.

        Args:
            lang (str): target language code.

        Returns:
            [type]: [description]
        """
        return pysbd.Segmenter(language=lang, clean=True)

    def _load_tts(
        self, tts_checkpoint: str, tts_config_path: str, use_cuda: bool
    ) -> None:
        """Load the TTS model.

        1. Load the model config.
        2. Init the model from the config.
        3. Load the model weights.
        4. Move the model to the GPU if CUDA is enabled.
        5. Init the speaker manager in the model.

        Args:
            tts_checkpoint (str): path to the model checkpoint.
            tts_config_path (str): path to the model config file.
            use_cuda (bool): enable/disable CUDA use.
        """
        # pylint: disable=global-statement
        self.tts_config = load_config(tts_config_path)
        if self.tts_config["use_phonemes"] and self.tts_config["phonemizer"] is None:
            raise ValueError("Phonemizer is not defined in the TTS config.")

        self.tts_model = setup_tts_model(config=self.tts_config)

        if not self.encoder_checkpoint:
            self._set_speaker_encoder_paths_from_tts_config()

        self.tts_model.load_checkpoint(self.tts_config, tts_checkpoint, eval=True)
        if use_cuda:
            self.tts_model.cuda()

        if self.encoder_checkpoint and hasattr(self.tts_model, "speaker_manager"):
            self.tts_model.speaker_manager.init_encoder(
                self.encoder_checkpoint, self.encoder_config, use_cuda
            )

    def _set_speaker_encoder_paths_from_tts_config(self):
        """Set the encoder paths from the tts model config for models with speaker encoders."""
        if hasattr(self.tts_config, "model_args") and hasattr(
            self.tts_config.model_args, "speaker_encoder_config_path"
        ):
            self.encoder_checkpoint = (
                self.tts_config.model_args.speaker_encoder_model_path
            )
            self.encoder_config = self.tts_config.model_args.speaker_encoder_config_path

    def _load_vocoder(self, model_file: str, model_config: str, use_cuda: bool) -> None:
        """Load the vocoder model.

        1. Load the vocoder config.
        2. Init the AudioProcessor for the vocoder.
        3. Init the vocoder model from the config.
        4. Move the model to the GPU if CUDA is enabled.

        Args:
            model_file (str): path to the model checkpoint.
            model_config (str): path to the model config file.
            use_cuda (bool): enable/disable CUDA use.
        """
        self.vocoder_config = load_config(model_config)
        self.vocoder_ap = AudioProcessor(verbose=False, **self.vocoder_config.audio)
        self.vocoder_model = setup_vocoder_model(self.vocoder_config)
        self.vocoder_model.load_checkpoint(self.vocoder_config, model_file, eval=True)
        if use_cuda:
            self.vocoder_model.cuda()

    def split_into_sentences(self, text) -> List[str]:
        """Split give text into sentences.

        Args:
            text (str): input text in string format.

        Returns:
            List[str]: list of sentences.
        """
        return self.seg.segment(text)

    def save_wav(self, wav: List[int], path: str) -> None:
        """Save the waveform as a file.

        Args:
            wav (List[int]): waveform as a list of values.
            path (str): output path to save the waveform.
        """
        wav = np.array(wav)
        self.tts_model.ap.save_wav(wav, path, self.output_sample_rate)

    def tts(
        self,
        text: str = "",
        speaker_name: str = "",
        language_name: str = "",
        speaker_wav=None,
        style_wav=None,
        style_text=None,
        reference_wav=None,
        reference_speaker_name=None,
    ) -> List[int]:
        """🐸 TTS magic. Run all the models and generate speech.

        Args:
            text (str): input text.
            speaker_name (str, optional): spekaer id for multi-speaker models. Defaults to "".
            language_name (str, optional): language id for multi-language models. Defaults to "".
            speaker_wav (Union[str, List[str]], optional): path to the speaker wav. Defaults to None.
            style_wav ([type], optional): style waveform for GST. Defaults to None.
            style_text ([type], optional): transcription of style_wav for Capacitron. Defaults to None.
            reference_wav ([type], optional): reference waveform for voice conversion. Defaults to None.
            reference_speaker_name ([type], optional): spekaer id of reference waveform. Defaults to None.
        Returns:
            List[int]: [description]
        """
        start_time = time.time()
        wavs = []

        if not text and not reference_wav:
            raise ValueError(
                "You need to define either `text` (for sythesis) or a `reference_wav` (for voice conversion) to use the Coqui TTS API."
            )

        if text:
            sens = self.split_into_sentences(text)
            print(" > Text splitted to sentences.")
            print(sens)

        # handle multi-speaker
        speaker_embedding = None
        speaker_id = None
        if self.tts_speakers_file or hasattr(
            self.tts_model.speaker_manager, "name_to_id"
        ):
            if speaker_name and isinstance(speaker_name, str):
                if self.tts_config.use_d_vector_file:
                    # get the average speaker embedding from the saved d_vectors.
                    speaker_embedding = (
                        self.tts_model.speaker_manager.get_mean_embedding(
                            speaker_name, num_samples=None, randomize=False
                        )
                    )
                    speaker_embedding = np.array(speaker_embedding)[
                        None, :
                    ]  # [1 x embedding_dim]
                else:
                    # get speaker idx from the speaker name
                    speaker_id = self.tts_model.speaker_manager.name_to_id[speaker_name]

            elif not speaker_name and not speaker_wav:
                raise ValueError(
                    " [!] Look like you use a multi-speaker model. "
                    "You need to define either a `speaker_name` or a `speaker_wav` to use a multi-speaker model."
                )
            else:
                speaker_embedding = None
        else:
            if speaker_name:
                raise ValueError(
                    f" [!] Missing speakers.json file path for selecting speaker {speaker_name}."
                    "Define path for speaker.json if it is a multi-speaker model or remove defined speaker idx. "
                )

        # handle multi-lingaul
        language_id = None
        if self.tts_languages_file or (
            hasattr(self.tts_model, "language_manager")
            and self.tts_model.language_manager is not None
        ):
            if language_name and isinstance(language_name, str):
                language_id = self.tts_model.language_manager.name_to_id[language_name]

            elif not language_name:
                raise ValueError(
                    " [!] Look like you use a multi-lingual model. "
                    "You need to define either a `language_name` or a `style_wav` to use a multi-lingual model."
                )

            else:
                raise ValueError(
                    f" [!] Missing language_ids.json file path for selecting language {language_name}."
                    "Define path for language_ids.json if it is a multi-lingual model or remove defined language idx. "
                )

        # compute a new d_vector from the given clip.
        if speaker_wav is not None:
            speaker_embedding = (
                self.tts_model.speaker_manager.compute_embedding_from_clip(speaker_wav)
            )

        use_gl = self.vocoder_model is None
        word_boundaries_list = []
        if not reference_wav:
            for sen in sens:
                # synthesize voice
                outputs = synthesis(
                    model=self.tts_model,
                    text=sen,
                    CONFIG=self.tts_config,
                    use_cuda=self.use_cuda,
                    speaker_id=speaker_id,
                    style_wav=style_wav,
                    style_text=style_text,
                    use_griffin_lim=use_gl,
                    d_vector=speaker_embedding,
                    language_id=language_id,
                )
                waveform = outputs["wav"]
                # import ipdb; ipdb.set_trace()
                mel_postnet_spec = (
                    outputs["outputs"]["model_outputs"][0].detach().cpu().numpy()
                )
                if not use_gl:
                    # denormalize tts output based on tts audio config
                    mel_postnet_spec = self.tts_model.ap.denormalize(
                        mel_postnet_spec.T
                    ).T
                    device_type = "cuda" if self.use_cuda else "cpu"
                    # renormalize spectrogram based on vocoder config
                    vocoder_input = self.vocoder_ap.normalize(mel_postnet_spec.T)
                    # compute scale factor for possible sample rate mismatch
                    scale_factor = [
                        1,
                        self.vocoder_config["audio"]["sample_rate"]
                        / self.tts_model.ap.sample_rate,
                    ]
                    if scale_factor[1] != 1:
                        print(" > interpolating tts model output.")
                        vocoder_input = interpolate_vocoder_input(
                            scale_factor, vocoder_input
                        )
                    else:
                        vocoder_input = torch.tensor(vocoder_input).unsqueeze(
                            0
                        )  # pylint: disable=not-callable
                    # run vocoder model
                    # [1, T, C]
                    waveform = self.vocoder_model.inference(
                        vocoder_input.to(device_type)
                    )
                if self.use_cuda and not use_gl:
                    waveform = waveform.cpu()
                if not use_gl:
                    waveform = waveform.numpy()
                waveform = waveform.squeeze()

                # trim silence
                if (
                    "do_trim_silence" in self.tts_config.audio
                    and self.tts_config.audio["do_trim_silence"]
                ):
                    waveform = trim_silence(waveform, self.tts_model.ap)

                wavs += list(waveform)
                wavs += [0] * 10000

                # compute stats
                audio_time = len(wavs) / self.tts_config.audio["sample_rate"]
                print(f" > Initial Audio duration: {audio_time}")

                word_boundaries_list.append(
                    self.compute_word_boundaries(
                        outputs["text_inputs"][0].tolist(), outputs["alignments"], audio_time
                    )
                )

        else:
            # get the speaker embedding or speaker id for the reference wav file
            reference_speaker_embedding = None
            reference_speaker_id = None
            if self.tts_speakers_file or hasattr(
                self.tts_model.speaker_manager, "name_to_id"
            ):
                if reference_speaker_name and isinstance(reference_speaker_name, str):
                    if self.tts_config.use_d_vector_file:
                        # get the speaker embedding from the saved d_vectors.
                        reference_speaker_embedding = (
                            self.tts_model.speaker_manager.get_embeddings_by_name(
                                reference_speaker_name
                            )[0]
                        )
                        reference_speaker_embedding = np.array(
                            reference_speaker_embedding
                        )[
                            None, :
                        ]  # [1 x embedding_dim]
                    else:
                        # get speaker idx from the speaker name
                        reference_speaker_id = (
                            self.tts_model.speaker_manager.name_to_id[
                                reference_speaker_name
                            ]
                        )
                else:
                    reference_speaker_embedding = (
                        self.tts_model.speaker_manager.compute_embedding_from_clip(
                            reference_wav
                        )
                    )
            outputs = transfer_voice(
                model=self.tts_model,
                CONFIG=self.tts_config,
                use_cuda=self.use_cuda,
                reference_wav=reference_wav,
                speaker_id=speaker_id,
                d_vector=speaker_embedding,
                use_griffin_lim=use_gl,
                reference_speaker_id=reference_speaker_id,
                reference_d_vector=reference_speaker_embedding,
            )
            waveform = outputs
            if not use_gl:
                mel_postnet_spec = outputs[0].detach().cpu().numpy()
                # denormalize tts output based on tts audio config
                mel_postnet_spec = self.tts_model.ap.denormalize(mel_postnet_spec.T).T
                device_type = "cuda" if self.use_cuda else "cpu"
                # renormalize spectrogram based on vocoder config
                vocoder_input = self.vocoder_ap.normalize(mel_postnet_spec.T)
                # compute scale factor for possible sample rate mismatch
                scale_factor = [
                    1,
                    self.vocoder_config["audio"]["sample_rate"]
                    / self.tts_model.ap.sample_rate,
                ]
                if scale_factor[1] != 1:
                    print(" > interpolating tts model output.")
                    vocoder_input = interpolate_vocoder_input(
                        scale_factor, vocoder_input
                    )
                else:
                    vocoder_input = torch.tensor(vocoder_input).unsqueeze(
                        0
                    )  # pylint: disable=not-callable
                # run vocoder model
                # [1, T, C]
                waveform = self.vocoder_model.inference(vocoder_input.to(device_type))
            if self.use_cuda:
                waveform = waveform.cpu()
            if not use_gl:
                waveform = waveform.numpy()
            wavs = waveform.squeeze()

        # compute stats
        process_time = time.time() - start_time
        audio_time = len(wavs) / self.tts_config.audio["sample_rate"]
        print(f" > Audio duration: {audio_time}")
        print(f" > Processing time: {process_time}")
        print(f" > Real-time factor: {process_time / audio_time}")

        # Merge word boundaries from all sentences
        word_boundaries = []
        current_text_offset = 0
        current_audio_offset = 0
        if len(word_boundaries_list) > 0:
            for word_boundaries_ in word_boundaries_list:
                for wb in word_boundaries_:
                    word_boundaries.append(
                        {
                            "text": wb["text"],
                            # "item": wb["item"],
                            # "text_offset": wb["text_offset"] + current_text_offset,
                            "offset": wb["offset"] + current_audio_offset,
                            # "word_length": wb["word_length"],
                        }
                    )
                # current_text_offset += word_boundaries_[-1]["text_offset"]
                current_audio_offset += (
                    word_boundaries_[-1]["offset"]
                    + 10000 / self.tts_config.audio["sample_rate"]
                )

        return wavs, word_boundaries

    def compute_word_boundaries(self, token_idx, alignments, audio_time):
        ret = []
        if alignments.dim() == 3:
            alignments = alignments[0]

        # To calculate elapsed seconds from the beginning of the audio:
        # seconds = # frames * hop_length / sample_rate

        # Get max index for each timestep
        max_idx = alignments.argmax(dim=0)
        # Get the time of each max index
        text_offset = 0

        last = 0.0
        for idx, token in zip(max_idx, token_idx):
            last = idx.item() * 1.0

        for idx, token in zip(max_idx, token_idx):
            audio_offset = ( 
                idx.item() * 1.0 / last * audio_time
                # AUDIO_OFFSET_RESOLUTION
                # * self.vocoder_ap.hop_length
                # * self.tts_config.audio["hop_length"]
                # / self.tts_config.audio["sample_rate"]
            )
            word = self.tts_model.tokenizer.ids_to_text([token])
            word_length = len(word)
            ret.append(
                {
                    "text": word,
                    "offset": audio_offset,
                    # "item": idx.item(),
                    # "text_offset": text_offset,
                    # "word_length": word_length,
                }
            )
            text_offset += word_length

        return ret


#########################################################################################################################################3

def synthesize_tts(
    text,
    output_path,
    model_name=DEFAULT_MODEL,
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
    # gst_style=None,
    # model_info_by_name=None,
    # model_info_by_idx=None,
    # list_speaker_idxs=False,
    # list_language_idxs=False,
    # save_spectogram=False,
    # progress_bar=True,
):

    # load model managera
    print("looking for model manager")
    print(  Path( TTS.__file__).parent )

    path = Path(TTS.__file__).parent / ".models.json"

    # manager = ModelManager(path, progress_bar=progress_bar)
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


#############################################################################################################################################################

class TtsService():
    """Speech service for Coqui TTS.
    See :func:`~manim_voiceover.services.coqui.synthesize.synthesize_tts`
    for more initialization options, e.g. Pass arguments defined in the synthesize_tts func as keyword arguments for setting different models."""

    def __init__(
        self,
        **kwargs,
    ):
        """"""
        self.init_kwargs = kwargs
        # prompt_ask_missing_package("TTS", "TTS")
        # SpeechService.__init__(self, **kwargs)

    def generate_from_text(
        self, text: str, cache_dir: str = None, path: str = None, **kwargs
    ) -> dict:
        """"""
        if cache_dir is None:
            cache_dir = self.cache_dir

        # input_text = remove_bookmarks(text)
        input_text = text
        input_data = {"input_text": text, "service": "coqui"}

        audio_path = fname

        if not kwargs:
            kwargs = self.init_kwargs

        _, word_boundaries = synthesize_tts(
            input_text, str(Path(cache_dir) / audio_path), **kwargs
        )

        json_dict = {
            "input_text": text,
            "input_data": input_data,
            "original_audio": audio_path,
            "word_boundaries": word_boundaries,
        }

        with open('boundaries.json', 'wt') as out:
           pprint(word_boundaries, stream=out)

        return json_dict


coqui = TtsService()




from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse
# import json
import uu
import base64

class MyRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        # Parse the query parameters
        parsed_path = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(parsed_path.query)

        # Check if 'query' parameter exists
        if 'query' in query:
            search_query = query['query'][0]

            # generate some text
            response = coqui.generate_from_text(search_query,"./")

            # Load WAV file from disk and uuencode it
            try:
                with open(fname, "rb") as f:
                    binary = f.read()
                    # response["binary"] = binary
                    # response["uuencoded"] = uu.encode(f, "output.uue")
                    response["base64"] = base64.b64encode(binary).decode('utf-8')
            except FileNotFoundError:
                response["error"] = "WAV file not found"

            with contextlib.closing(wave.open(fname,'r')) as f:
                frames = f.getnframes()
                rate = f.getframerate()
                response["frames"] = frames
                response["rate"] = rate
                response["duration"] = frames / rate

            # build pretty json
            response_json = json.dumps(response, indent=4)  # Pretty print JSON

            # Send response
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin","*")
            self.send_header("Access-Control-Allow-Credentials","true")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Origin, Content-Type, Accept")

            self.end_headers()
            self.wfile.write(response_json.encode())
        else:
            # If 'query' parameter is missing, return a 400 Bad Request
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"Bad Request: 'query' parameter is missing")

def run(server_class=HTTPServer, handler_class=MyRequestHandler, addr='localhost', port=8000):
    server_address = (addr, port)
    httpd = server_class(server_address, handler_class)
    print(f"Starting server on port {port}...")
    httpd.serve_forever()

run()

# response = coqui.generate_from_text("this is a very long single test that is a good thing","./")
# pprint(response)

