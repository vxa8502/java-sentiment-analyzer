package sentiment.preprocessing;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive unit tests for ContractionExpander.
 *
 * Tests cover:
 * - Negative contractions expansion
 * - Positive contractions expansion
 * - Capitalization preservation
 * - Edge cases and validation
 * - Utility methods
 */
@DisplayName("ContractionExpander Unit Tests")
class ContractionExpanderTest {

    private ContractionExpander expander;

    @BeforeEach
    void setUp() {
        expander = new ContractionExpander();
    }

    // ==================== NEGATIVE CONTRACTIONS TESTS ====================

    @Test
    @DisplayName("expand should expand don't to do not")
    void testExpand_DontToDoNot() {
        String result = expander.expand("I don't like this");
        assertTrue(result.contains("do not"));
        assertFalse(result.contains("don't"));
    }

    @Test
    @DisplayName("expand should expand doesn't to does not")
    void testExpand_DoesntToDoesNot() {
        String result = expander.expand("It doesn't work");
        assertTrue(result.contains("does not"));
    }

    @Test
    @DisplayName("expand should expand didn't to did not")
    void testExpand_DidntToDidNot() {
        String result = expander.expand("I didn't see it");
        assertTrue(result.contains("did not"));
    }

    @Test
    @DisplayName("expand should expand won't to will not")
    void testExpand_WontToWillNot() {
        String result = expander.expand("I won't go");
        assertTrue(result.contains("will not"));
    }

    @Test
    @DisplayName("expand should expand wouldn't to would not")
    void testExpand_WouldntToWouldNot() {
        String result = expander.expand("She wouldn't say");
        assertTrue(result.contains("would not"));
    }

    @Test
    @DisplayName("expand should expand can't to cannot")
    void testExpand_CantToCannot() {
        String result = expander.expand("I can't do it");
        assertTrue(result.contains("cannot"));
    }

    @Test
    @DisplayName("expand should expand couldn't to could not")
    void testExpand_CouldntToCouldNot() {
        String result = expander.expand("He couldn't find it");
        assertTrue(result.contains("could not"));
    }

    @Test
    @DisplayName("expand should expand shouldn't to should not")
    void testExpand_ShouldntToShouldNot() {
        String result = expander.expand("You shouldn't worry");
        assertTrue(result.contains("should not"));
    }

    @Test
    @DisplayName("expand should expand mustn't to must not")
    void testExpand_MustntToMustNot() {
        String result = expander.expand("We mustn't forget");
        assertTrue(result.contains("must not"));
    }

    @Test
    @DisplayName("expand should expand isn't to is not")
    void testExpand_IsntToIsNot() {
        String result = expander.expand("It isn't fair");
        assertTrue(result.contains("is not"));
    }

    @Test
    @DisplayName("expand should expand aren't to are not")
    void testExpand_ArentToAreNot() {
        String result = expander.expand("They aren't here");
        assertTrue(result.contains("are not"));
    }

    @Test
    @DisplayName("expand should expand wasn't to was not")
    void testExpand_WasntToWasNot() {
        String result = expander.expand("He wasn't there");
        assertTrue(result.contains("was not"));
    }

    @Test
    @DisplayName("expand should expand weren't to were not")
    void testExpand_WerentToWereNot() {
        String result = expander.expand("We weren't sure");
        assertTrue(result.contains("were not"));
    }

    @Test
    @DisplayName("expand should expand haven't to have not")
    void testExpand_HaventToHaveNot() {
        String result = expander.expand("I haven't finished");
        assertTrue(result.contains("have not"));
    }

    @Test
    @DisplayName("expand should expand hasn't to has not")
    void testExpand_HasntToHasNot() {
        String result = expander.expand("She hasn't arrived");
        assertTrue(result.contains("has not"));
    }

    @Test
    @DisplayName("expand should expand hadn't to had not")
    void testExpand_HadntToHadNot() {
        String result = expander.expand("They hadn't expected");
        assertTrue(result.contains("had not"));
    }

    // ==================== POSITIVE CONTRACTIONS TESTS ====================

    @Test
    @DisplayName("expand should expand I'm to I am")
    void testExpand_ImToIAm() {
        String result = expander.expand("I'm happy");
        assertTrue(result.contains("I am"));
    }

    @Test
    @DisplayName("expand should expand you're to you are")
    void testExpand_YoureToYouAre() {
        String result = expander.expand("You're right");
        assertTrue(result.toLowerCase().contains("you are"));
    }

    @Test
    @DisplayName("expand should expand he's to he is")
    void testExpand_HesToHeIs() {
        String result = expander.expand("He's coming");
        assertTrue(result.toLowerCase().contains("he is"));
    }

    @Test
    @DisplayName("expand should expand she's to she is")
    void testExpand_ShesToSheIs() {
        String result = expander.expand("She's here");
        assertTrue(result.toLowerCase().contains("she is"));
    }

    @Test
    @DisplayName("expand should expand it's to it is")
    void testExpand_ItsToItIs() {
        String result = expander.expand("It's great");
        assertTrue(result.toLowerCase().contains("it is"));
    }

    @Test
    @DisplayName("expand should expand we're to we are")
    void testExpand_WereToWeAre() {
        String result = expander.expand("We're ready");
        assertTrue(result.toLowerCase().contains("we are"));
    }

    @Test
    @DisplayName("expand should expand they're to they are")
    void testExpand_TheyreToTheyAre() {
        String result = expander.expand("They're coming");
        assertTrue(result.toLowerCase().contains("they are"));
    }

    @Test
    @DisplayName("expand should expand I've to I have")
    void testExpand_IveToIHave() {
        String result = expander.expand("I've finished");
        assertTrue(result.contains("I have"));
    }

    @Test
    @DisplayName("expand should expand I'll to I will")
    void testExpand_IllToIWill() {
        String result = expander.expand("I'll be there");
        assertTrue(result.contains("I will"));
    }

    @Test
    @DisplayName("expand should expand I'd to I would")
    void testExpand_IdToIWould() {
        String result = expander.expand("I'd love to");
        assertTrue(result.contains("I would"));
    }

    @Test
    @DisplayName("expand should expand let's to let us")
    void testExpand_LetsToLetUs() {
        String result = expander.expand("Let's go");
        assertTrue(result.toLowerCase().contains("let us"));
    }

    // ==================== INFORMAL CONTRACTIONS TESTS ====================

    @Test
    @DisplayName("expand should expand ain't to am not")
    void testExpand_AintToAmNot() {
        String result = expander.expand("I ain't going");
        assertTrue(result.toLowerCase().contains("am not"));
    }

    @Test
    @DisplayName("expand should expand y'all to you all")
    void testExpand_YallToYouAll() {
        String result = expander.expand("Y'all come back");
        assertTrue(result.toLowerCase().contains("you all"));
    }

    // ==================== CAPITALIZATION TESTS ====================

    @Test
    @DisplayName("expand should preserve lowercase")
    void testExpand_PreservesLowercase() {
        String result = expander.expand("i don't know");
        assertTrue(result.contains("do not"));
        assertFalse(result.contains("Do not"));
    }

    @Test
    @DisplayName("expand should preserve first letter capitalization")
    void testExpand_PreservesCapitalization() {
        String result = expander.expand("Don't worry");
        assertTrue(result.contains("Do not"));
    }

    @Test
    @DisplayName("expand should preserve all caps")
    void testExpand_PreservesAllCaps() {
        String result = expander.expand("I DON'T KNOW");
        assertTrue(result.contains("DO NOT"));
    }

    @Test
    @DisplayName("expand should handle mixed case sentences")
    void testExpand_HandlesMixedCase() {
        String result = expander.expand("I Don't Think So");
        assertTrue(result.toLowerCase().contains("do not"));
    }

    // ==================== MULTIPLE CONTRACTIONS TESTS ====================

    @Test
    @DisplayName("expand should handle multiple contractions in one sentence")
    void testExpand_MultipleContractions() {
        String result = expander.expand("I don't think it's right");
        assertTrue(result.contains("do not"));
        assertTrue(result.contains("it is"));
    }

    @Test
    @DisplayName("expand should handle text with many contractions")
    void testExpand_ManyContractions() {
        String text = "I don't think she's right, and he won't agree";
        String result = expander.expand(text);

        assertTrue(result.contains("do not"));
        assertTrue(result.contains("she is"));
        assertTrue(result.contains("will not"));
    }

    // ==================== EDGE CASES TESTS ====================

    @Test
    @DisplayName("expand should handle null input")
    void testExpand_NullInput() {
        String result = expander.expand(null);
        assertEquals("", result);
    }

    @Test
    @DisplayName("expand should handle empty input")
    void testExpand_EmptyInput() {
        String result = expander.expand("");
        assertEquals("", result);
    }

    @Test
    @DisplayName("expand should handle whitespace-only input")
    void testExpand_WhitespaceOnly() {
        String result = expander.expand("   ");
        assertEquals("   ", result);
    }

    @Test
    @DisplayName("expand should handle text without contractions")
    void testExpand_NoContractions() {
        String text = "This is a normal sentence";
        String result = expander.expand(text);
        assertEquals(text, result);
    }

    @Test
    @DisplayName("expand should handle contractions at start of text")
    void testExpand_ContractionAtStart() {
        String result = expander.expand("Don't do that");
        assertTrue(result.startsWith("Do not"));
    }

    @Test
    @DisplayName("expand should handle contractions at end of text")
    void testExpand_ContractionAtEnd() {
        String result = expander.expand("I know he can't");
        assertTrue(result.endsWith("cannot"));
    }

    @Test
    @DisplayName("expand should use word boundaries to avoid partial matches")
    void testExpand_WordBoundaries() {
        String text = "The candy doesn't melt";
        String result = expander.expand(text);

        assertTrue(result.contains("candy"));
        assertTrue(result.contains("does not"));
    }

    // ==================== UTILITY METHODS TESTS ====================

    @Test
    @DisplayName("hasContractions should return true when contractions present")
    void testHasContractions_True() {
        assertTrue(expander.hasContractions("I don't know"));
        assertTrue(expander.hasContractions("She's here"));
        assertTrue(expander.hasContractions("We can't go"));
    }

    @Test
    @DisplayName("hasContractions should return false when no contractions")
    void testHasContractions_False() {
        assertFalse(expander.hasContractions("This is normal text"));
        assertFalse(expander.hasContractions("No contractions here"));
    }

    @Test
    @DisplayName("hasContractions should handle null input")
    void testHasContractions_Null() {
        assertFalse(expander.hasContractions(null));
    }

    @Test
    @DisplayName("hasContractions should handle empty input")
    void testHasContractions_Empty() {
        assertFalse(expander.hasContractions(""));
        assertFalse(expander.hasContractions("   "));
    }

    @Test
    @DisplayName("countContractions should count correctly")
    void testCountContractions() {
        assertEquals(0, expander.countContractions("No contractions"));
        assertEquals(1, expander.countContractions("I don't know"));
        assertEquals(2, expander.countContractions("I don't think it's right"));
        assertEquals(3, expander.countContractions("I don't think she's right and he won't agree"));
    }

    @Test
    @DisplayName("countContractions should handle null input")
    void testCountContractions_Null() {
        assertEquals(0, expander.countContractions(null));
    }

    @Test
    @DisplayName("countContractions should handle empty input")
    void testCountContractions_Empty() {
        assertEquals(0, expander.countContractions(""));
        assertEquals(0, expander.countContractions("   "));
    }

    @Test
    @DisplayName("getStats should return contraction statistics")
    void testGetStats() {
        ContractionExpander.ContractionStats stats = expander.getStats();

        assertNotNull(stats);
        assertTrue(stats.totalContractions() > 0);
        assertTrue(stats.negativeContractions() > 0);
        assertTrue(stats.negativeContractions() <= stats.totalContractions());
    }

    @Test
    @DisplayName("getVersion should return version string")
    void testGetVersion() {
        String version = expander.getVersion();

        assertNotNull(version);
        assertFalse(version.isEmpty());
    }

    // ==================== INTEGRATION TESTS ====================

    @Test
    @DisplayName("expand should handle complex real-world sentence")
    void testExpand_ComplexSentence() {
        String text = "I don't think it's right that she won't come, but I can't force her";
        String result = expander.expand(text);

        assertTrue(result.contains("do not"));
        assertTrue(result.contains("it is"));
        assertTrue(result.contains("will not"));
        assertTrue(result.contains("cannot"));
        assertFalse(result.contains("don't"));
        assertFalse(result.contains("it's"));
        assertFalse(result.contains("won't"));
        assertFalse(result.contains("can't"));
    }

    @Test
    @DisplayName("expand should handle text with punctuation")
    void testExpand_WithPunctuation() {
        String text = "I don't know! She's here? We can't believe it.";
        String result = expander.expand(text);

        assertTrue(result.toLowerCase().contains("do not"));
        assertTrue(result.toLowerCase().contains("she is"));
        assertTrue(result.toLowerCase().contains("cannot"));
        assertTrue(result.contains("!"));
        assertTrue(result.contains("?"));
        assertTrue(result.contains("."));
    }
}
