#@title Imports and Definitions

import functools
import os

import numpy as np
import tensorflow.compat.v2 as tf

import functools
import gin
import jax
import librosa
import note_seq
import seqio
import t5
import t5x

from mt3 import metrics_utils
from mt3 import models
from mt3 import network
from mt3 import note_sequences
from mt3 import preprocessors
from mt3 import spectrograms
from mt3 import vocabularies



import nest_asyncio
nest_asyncio.apply()


def parent_dir_and_name(file_path):
    """
    >>> file_path="a/b.c"
    >>> parent_dir_and_name(file_path)
    ('/root/.../a', 'b.c')
    :param file_path:
    :return:
    """
    return os.path.split(os.path.abspath(file_path))


def basename_and_extension(file_path):
    """
    >>> file_path="a/b.c"
    >>> basename_and_extension(file_path)
    ('b', '.c')
    :param file_path:
    :return:
    """
    return os.path.splitext(os.path.basename(file_path))


def get_things_in_loc(in_path, just_files=True, endswith=None):
    """
    in_path can be file path or dir path.
    This function return a list of file paths
    in in_path if in_path is a dir, or within the
    parent path of in_path if it is not a dir.
    just_files=False will let the function go recursively
    into the subdirs.
    :endswith: None or a list of file extensions (to end with).
    """
    # TODO: check for file
    if not os.path.exists(in_path):
        print(str(in_path) + " does not exists!")
        return
    re_list = []
    if not os.path.isdir(in_path):
        in_path = parent_dir_and_name(in_path)[0]

    for name in os.listdir(in_path):
        name_path = os.path.abspath(os.path.join(in_path, name))
        if os.path.isfile(name_path) and (endswith is None or (True in [name_path.endswith(ext) for ext in endswith])):
            re_list.append(name_path)
        elif not just_files:
            if os.path.isdir(name_path):
                re_list += get_things_in_loc(name_path, just_files=just_files, endswith=endswith)
    return re_list




SAMPLE_RATE = 16000
SF2_PATH = 'SGM-v2.01-Sal-Guit-Bass-V1.3.sf2'
MODEL = 'mt3' # 'ismir2021' is piano only, 'mt3' is multi-instrument
GIN_FILES_PATH = 'mt3/gin'
CHECKPOINT_PATH = 'checkpoints'
SOURCE_PATH = '../audio_files'
DEST_PATH = '../midi_files'
# gin_files = ['/content/mt3/gin/model.gin',
#               f'/content/mt3/gin/{model_type}.gin']





class InferenceModel(object):
    """Wrapper of T5X model for music transcription."""

    def __init__(self, checkpoint_path, model_type="mt3"):

        # Model Constants.
        if model_type == "ismir2021":
            num_velocity_bins = 127
            self.encoding_spec = note_sequences.NoteEncodingSpec
            self.inputs_length = 512
        elif model_type == "mt3":
            num_velocity_bins = 1
            self.encoding_spec = note_sequences.NoteEncodingWithTiesSpec
            self.inputs_length = 256
        else:
            raise ValueError("unknown model_type: %s" % model_type)

        gin_files = ["{GIN_FILES_PATH}/model.gin", f"{GIN_FILES_PATH}/{model_type}.gin"]

        self.batch_size = 8
        self.outputs_length = 1024
        self.sequence_length = {
            "inputs": self.inputs_length,
            "targets": self.outputs_length,
        }

        self.partitioner = t5x.partitioning.PjitPartitioner(num_partitions=1)

        # Build Codecs and Vocabularies.
        self.spectrogram_config = spectrograms.SpectrogramConfig()
        self.codec = vocabularies.build_codec(
            vocab_config=vocabularies.VocabularyConfig(
                num_velocity_bins=num_velocity_bins
            )
        )
        self.vocabulary = vocabularies.vocabulary_from_codec(self.codec)
        self.output_features = {
            "inputs": seqio.ContinuousFeature(dtype=tf.float32, rank=2),
            "targets": seqio.Feature(vocabulary=self.vocabulary),
        }

        # Create a T5X model.
        self._parse_gin(gin_files)
        self.model = self._load_model()

        # Restore from checkpoint.
        self.restore_from_checkpoint(checkpoint_path)

    @property
    def input_shapes(self):
        return {
            "encoder_input_tokens": (self.batch_size, self.inputs_length),
            "decoder_input_tokens": (self.batch_size, self.outputs_length),
        }

    def _parse_gin(self, gin_files):
        """Parse gin files used to train the model."""
        gin_bindings = [
            "from __gin__ import dynamic_registration",
            "from mt3 import vocabularies",
            "VOCAB_CONFIG=@vocabularies.VocabularyConfig()",
            "vocabularies.VocabularyConfig.num_velocity_bins=%NUM_VELOCITY_BINS",
        ]
        with gin.unlock_config():
            gin.parse_config_files_and_bindings(
                gin_files, gin_bindings, finalize_config=False
            )

    def _load_model(self):
        """Load up a T5X `Model` after parsing training gin config."""
        model_config = gin.get_configurable(network.T5Config)()
        module = network.Transformer(config=model_config)
        return models.ContinuousInputsEncoderDecoderModel(
            module=module,
            input_vocabulary=self.output_features["inputs"].vocabulary,
            output_vocabulary=self.output_features["targets"].vocabulary,
            optimizer_def=t5x.adafactor.Adafactor(decay_rate=0.8, step_offset=0),
            input_depth=spectrograms.input_depth(self.spectrogram_config),
        )

    def restore_from_checkpoint(self, checkpoint_path):
        """Restore training state from checkpoint, resets self._predict_fn()."""
        train_state_initializer = t5x.utils.TrainStateInitializer(
            optimizer_def=self.model.optimizer_def,
            init_fn=self.model.get_initial_variables,
            input_shapes=self.input_shapes,
            partitioner=self.partitioner,
        )

        restore_checkpoint_cfg = t5x.utils.RestoreCheckpointConfig(
            path=checkpoint_path, mode="specific", dtype="float32"
        )

        train_state_axes = train_state_initializer.train_state_axes
        self._predict_fn = self._get_predict_fn(train_state_axes)
        self._train_state = train_state_initializer.from_checkpoint_or_scratch(
            [restore_checkpoint_cfg], init_rng=jax.random.PRNGKey(0)
        )

    @functools.lru_cache()
    def _get_predict_fn(self, train_state_axes):
        """Generate a partitioned prediction function for decoding."""

        def partial_predict_fn(params, batch, decode_rng):
            return self.model.predict_batch_with_aux(
                params, batch, decoder_params={"decode_rng": None}
            )

        return self.partitioner.partition(
            partial_predict_fn,
            in_axis_resources=(
                train_state_axes.params,
                t5x.partitioning.PartitionSpec(
                    "data",
                ),
                None,
            ),
            out_axis_resources=t5x.partitioning.PartitionSpec(
                "data",
            ),
        )

    def predict_tokens(self, batch, seed=0):
        """Predict tokens from preprocessed dataset batch."""
        prediction, _ = self._predict_fn(
            self._train_state.params, batch, jax.random.PRNGKey(seed)
        )
        return self.vocabulary.decode_tf(prediction).numpy()

    def __call__(self, audio):
        """Infer note sequence from audio samples.

        Args:
          audio: 1-d numpy array of audio samples (16kHz) for a single example.

        Returns:
          A note_sequence of the transcribed audio.
        """
        ds = self.audio_to_dataset(audio)
        ds = self.preprocess(ds)

        model_ds = self.model.FEATURE_CONVERTER_CLS(pack=False)(
            ds, task_feature_lengths=self.sequence_length
        )
        model_ds = model_ds.batch(self.batch_size)

        inferences = (
            tokens
            for batch in model_ds.as_numpy_iterator()
            for tokens in self.predict_tokens(batch)
        )

        predictions = []
        for example, tokens in zip(ds.as_numpy_iterator(), inferences):
            predictions.append(self.postprocess(tokens, example))

        result = metrics_utils.event_predictions_to_ns(
            predictions, codec=self.codec, encoding_spec=self.encoding_spec
        )
        return result["est_ns"]

    def audio_to_dataset(self, audio):
        """Create a TF Dataset of spectrograms from input audio."""
        frames, frame_times = self._audio_to_frames(audio)
        return tf.data.Dataset.from_tensors(
            {
                "inputs": frames,
                "input_times": frame_times,
            }
        )

    def _audio_to_frames(self, audio):
        """Compute spectrogram frames from audio."""
        frame_size = self.spectrogram_config.hop_width
        padding = [0, frame_size - len(audio) % frame_size]
        audio = np.pad(audio, padding, mode="constant")
        frames = spectrograms.split_audio(audio, self.spectrogram_config)
        num_frames = len(audio) // frame_size
        times = np.arange(num_frames) / self.spectrogram_config.frames_per_second
        return frames, times

    def preprocess(self, ds):
        pp_chain = [
            functools.partial(
                t5.data.preprocessors.split_tokens_to_inputs_length,
                sequence_length=self.sequence_length,
                output_features=self.output_features,
                feature_key="inputs",
                additional_feature_keys=["input_times"],
            ),
            # Cache occurs here during training.
            preprocessors.add_dummy_targets,
            functools.partial(
                preprocessors.compute_spectrograms,
                spectrogram_config=self.spectrogram_config,
            ),
        ]
        for pp in pp_chain:
            ds = pp(ds)
        return ds

    def postprocess(self, tokens, example):
        tokens = self._trim_eos(tokens)
        start_time = example["input_times"][0]
        # Round down to nearest symbolic token step.
        start_time -= start_time % (1 / self.codec.steps_per_second)
        return {
            "est_tokens": tokens,
            "start_time": start_time,
            # Internal MT3 code expects raw inputs, not used here.
            "raw_inputs": [],
        }

    @staticmethod
    def _trim_eos(tokens):
        tokens = np.array(tokens, np.int32)
        if vocabularies.DECODED_EOS_ID in tokens:
            tokens = tokens[: np.argmax(tokens == vocabularies.DECODED_EOS_ID)]
        return tokens


inference_model = InferenceModel(checkpoint_path = f'{CHECKPOINT_PATH}/{MODEL}/', model_type = MODEL)

for file_path in get_things_in_loc(SOURCE_PATH, just_files=True, endswith=['.wav']):
    print(file_path)
    

    # process data
    audio = note_seq.audio_io.wav_data_to_samples_librosa(
            file_path, sample_rate=SAMPLE_RATE
        )
    # note_seq.notebook_utils.colab_play(audio, sample_rate=SAMPLE_RATE)

    # transcribe audio

    est_ns = inference_model(audio)

    # note_seq.play_sequence(est_ns, synth=note_seq.fluidsynth,
    #                        sample_rate=SAMPLE_RATE, sf2_path=SF2_PATH)
    # note_seq.plot_sequence(est_ns)

    # convert to midi
    dest_file_name = basename_and_extension(parent_dir_and_name(file_path)[1])[0]
    note_seq.sequence_proto_to_midi_file(est_ns, f'{DEST_PATH}/{dest_file_name}.midi')