import pretty_midi as pm
from PIL import Image

def compareMidi(goldMIDIPath, predMIDIPath):
    PIXPERPITCH = 15
    MINPITCH = 21
    MAXPITCH = 108
    PIXPERSEC = 100
    SPACEPRED = 4
    PREDCOLOR = (0,0,255)
    GOLDCOLOR = (255,0,0)
    PitchRange = MAXPITCH - MINPITCH + 1
    Pitchpixs = (PitchRange+1)*PIXPERPITCH
    def pitch2pix(pitch):
        return Pitchpixs-((pitch-MINPITCH+1)*PIXPERPITCH)
    def time2pix(sec):
        return int(sec*PIXPERSEC)
    def setOutput(output, note, pred):
        for i in range(time2pix(note.start),time2pix(note.end)+1):
            output[pitch2pix(note.pitch) + (SPACEPRED if pred else 0)][i] = PREDCOLOR if pred else GOLDCOLOR
            output[pitch2pix(note.pitch) - (SPACEPRED if pred else 0)][i] = PREDCOLOR if pred else GOLDCOLOR
        for i in range(-SPACEPRED,SPACEPRED+1):
            output[i+pitch2pix(note.pitch)][time2pix(note.start)] = PREDCOLOR if pred else GOLDCOLOR
            output[i+pitch2pix(note.pitch)][time2pix(note.end)] = PREDCOLOR if pred else GOLDCOLOR
    
    goldMID = pm.PrettyMIDI(goldMIDIPath)
    predMID = pm.PrettyMIDI(predMIDIPath)
    
    end = max(goldMID.get_end_time(),predMID.get_end_time())
    length = int((end+1)*PIXPERSEC)
    im = Image.new("RGB",(length,Pitchpixs))
    output = [[(255,255,255) for y in range(length)] for x in range(Pitchpixs)]
    
    #reference pitch-lines
    for i in range(length):
        for j in [43,47,50,53,57,64,67,71,74,77]:
            output[pitch2pix(j)][i] = (0,255,0)
            
    #reference time-lines:
    for i in range(Pitchpixs):
        for j in range(0,int(length/PIXPERSEC)):
            output[i][time2pix(j)] = (240,240,240)
    
    for inst in predMID.instruments:
        for note in inst.notes:
            setOutput(output, note, True)
                    
    for inst in goldMID.instruments:
        for note in inst.notes:
            setOutput(output, note, False)
            
    im.putdata([item for sublist in output for item in sublist])
    im.save('test.PNG')
	
compareMidi("gold.mid","pred.mid")