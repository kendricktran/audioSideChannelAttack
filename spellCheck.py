from spellchecker import SpellChecker
import sys

def spell_check_predictions(predictions):
    """
    Takes a string of predictions and returns the spell-corrected version.
    
    Args:
        predictions (str): String of predicted characters
        
    Returns:
        str: Spell-corrected version of the predictions
    """
    # Initialize spell checker
    spell = SpellChecker()
    
    # Split into words and correct each word
    words = predictions.split('_')
    corrected_words = []
    
    for word in words:
        if word:  # Skip empty strings
            corrected = spell.correction(word)
            corrected_words.append(corrected if corrected else word)
    
    return '_'.join(corrected_words)

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 spellCheck.py <predictions>")
        print("Example: python3 spellCheck.py 'the_quik_brown_fox'")
        sys.exit(1)
        
    predictions = sys.argv[1]
    print("Original Prediction:", predictions)
    corrected = spell_check_predictions(predictions)
    print("Spell-Corrected Prediction:", corrected)

if __name__ == "__main__":
    main() 