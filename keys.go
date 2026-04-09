package story

import (
	"fmt"

	"github.com/google/uuid"
)

// Key schema (all keys are ASCII, lexicographically orderable):
//
//   c:state                        — calibrator state (σ_global, dimensionality, last batch)
//   s:{storyID}:m                  — story metadata
//   s:{storyID}:s:{signalID}       — signal data belonging to a story
//   o:{signalID}                   — outlier signal (not yet assigned to a story)
//   t:{unix_sec_10d}:{storyID}     — time index for Tier 3 range scans

func keyCalibState() []byte {
	return []byte("c:state")
}

func keyStoryMeta(storyID uuid.UUID) []byte {
	return fmt.Appendf(nil, "s:%s:m", storyID)
}

// keyStoryPrefix returns the prefix covering all keys for a story
// (metadata + signals): "s:{storyID}:".
func keyStoryPrefix(storyID uuid.UUID) []byte {
	return fmt.Appendf(nil, "s:%s:", storyID)
}

func keySignal(storyID, signalID uuid.UUID) []byte {
	return fmt.Appendf(nil, "s:%s:s:%s", storyID, signalID)
}

// keySignalPrefix returns the prefix covering all signal keys for a story:
// "s:{storyID}:s:".
func keySignalPrefix(storyID uuid.UUID) []byte {
	return fmt.Appendf(nil, "s:%s:s:", storyID)
}

func keyOutlier(signalID uuid.UUID) []byte {
	return fmt.Appendf(nil, "o:%s", signalID)
}

// keyTimeIndex returns a time-index key for the given story.
// unixSec is zero-padded to 10 digits (sufficient until year 2286) so keys
// sort lexicographically by time.
func keyTimeIndex(unixSec int64, storyID uuid.UUID) []byte {
	return fmt.Appendf(nil, "t:%010d:%s", unixSec, storyID)
}

// keyTimeIndexFrom returns the lower bound for a range scan starting at unixSec.
func keyTimeIndexFrom(unixSec int64) []byte {
	return fmt.Appendf(nil, "t:%010d:", unixSec)
}
