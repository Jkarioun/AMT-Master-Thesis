import pretty_midi as pm
from PIL import Image


def compare_midi(gold_MIDI_path, pred_MIDI_path):
    PIX_PER_PITCH = 15
    MIN_PITCH = 21
    MAX_PITCH = 108
    PIX_PER_SEC = 100
    SPACE_PRED = 4
    PRED_COLOR = (0, 0, 255)
    GOLD_COLOR = (255, 0, 0)
    pitch_range = MAX_PITCH - MIN_PITCH + 1
    pitch_pixs = (pitch_range + 1) * PIX_PER_PITCH

    def pitch2pix(pitch):
        return pitch_pixs - ((pitch - MIN_PITCH + 1) * PIX_PER_PITCH)

    def time2pix(sec):
        return int(sec * PIX_PER_SEC)

    def set_output(outputs, note, pred):
        for i in range(time2pix(note.start), time2pix(note.end) + 1):
            outputs[pitch2pix(note.pitch) + (SPACE_PRED if pred else 0)][i] = PRED_COLOR if pred else GOLD_COLOR
            outputs[pitch2pix(note.pitch) - (SPACE_PRED if pred else 0)][i] = PRED_COLOR if pred else GOLD_COLOR
        for i in range(-SPACE_PRED, SPACE_PRED + 1):
            outputs[i + pitch2pix(note.pitch)][time2pix(note.start)] = PRED_COLOR if pred else GOLD_COLOR
            outputs[i + pitch2pix(note.pitch)][time2pix(note.end)] = PRED_COLOR if pred else GOLD_COLOR

    gold_MID = pm.PrettyMIDI(gold_MIDI_path)
    pred_MID = pm.PrettyMIDI(pred_MIDI_path)

    end = max(gold_MID.get_end_time(), pred_MID.get_end_time())
    length = int((end + 1) * PIX_PER_SEC)
    im = Image.new("RGB", (length, pitch_pixs))
    output = [[(255, 255, 255) for _ in range(length)] for _ in range(pitch_pixs)]

    # reference pitch-lines
    for i in range(length):
        for j in [43, 47, 50, 53, 57, 64, 67, 71, 74, 77]:
            output[pitch2pix(j)][i] = (0, 255, 0)

    # reference time-lines:
    for i in range(pitch_pixs):
        for j in range(0, int(length / PIX_PER_SEC)):
            output[i][time2pix(j)] = (240, 240, 240)

    for inst in pred_MID.instruments:
        for note in inst.notes:
            set_output(output, note, True)

    for inst in gold_MID.instruments:
        for note in inst.notes:
            set_output(output, note, False)

    im.putdata([item for sublist in output for item in sublist])
    im.save('test.PNG')


gold_MID = pm.PrettyMIDI("bug.mid")

#compare_midi("bug.mid", "bug.mid")
