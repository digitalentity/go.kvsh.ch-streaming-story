package story

import (
	"bytes"
	"testing"

	"github.com/google/uuid"
	"github.com/stretchr/testify/assert"
)

var (
	keysTestStoryID  = uuid.MustParse("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
	keysTestSignalID = uuid.MustParse("11111111-2222-3333-4444-555555555555")
)

func TestKeyCalibState(t *testing.T) {
	assert.Equal(t, []byte("c:state"), keyCalibState())
}

func TestKeyStoryMeta(t *testing.T) {
	assert.Equal(t,
		[]byte("s:aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee:m"),
		keyStoryMeta(keysTestStoryID),
	)
}

func TestKeyStoryPrefix(t *testing.T) {
	assert.Equal(t,
		[]byte("s:aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee:"),
		keyStoryPrefix(keysTestStoryID),
	)
}

func TestKeySignal(t *testing.T) {
	assert.Equal(t,
		[]byte("s:aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee:s:11111111-2222-3333-4444-555555555555"),
		keySignal(keysTestStoryID, keysTestSignalID),
	)
}

func TestKeySignalPrefix(t *testing.T) {
	assert.Equal(t,
		[]byte("s:aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee:s:"),
		keySignalPrefix(keysTestStoryID),
	)
}

func TestKeyOutlier(t *testing.T) {
	assert.Equal(t,
		[]byte("o:11111111-2222-3333-4444-555555555555"),
		keyOutlier(keysTestSignalID),
	)
}

func TestKeyTimeIndex(t *testing.T) {
	t.Run("exact_format", func(t *testing.T) {
		assert.Equal(t,
			[]byte("t:1234567890:aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"),
			keyTimeIndex(1234567890, keysTestStoryID),
		)
	})

	t.Run("zero_pads_to_10_digits", func(t *testing.T) {
		assert.Equal(t,
			[]byte("t:0000000001:aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"),
			keyTimeIndex(1, keysTestStoryID),
		)
	})

	t.Run("lexicographic_order_matches_chronological_order", func(t *testing.T) {
		earlier := keyTimeIndex(1_000_000_000, keysTestStoryID)
		later := keyTimeIndex(2_000_000_000, keysTestStoryID)
		assert.Negative(t, bytes.Compare(earlier, later),
			"earlier timestamp must produce a key that sorts before later timestamp")
	})
}

func TestKeyTimeIndexFrom(t *testing.T) {
	assert.Equal(t, []byte("t:1234567890:"), keyTimeIndexFrom(1234567890))
}
