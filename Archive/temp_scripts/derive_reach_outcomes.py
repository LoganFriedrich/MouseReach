"""
Derive individual reach outcomes from segment data.

Reach Outcome Categories:
    CAUSAL REACHES (the reach during which interaction_frame occurred):
        - retrieved: Grabbed the pellet
        - displaced_sa: Knocked pellet into scoring area
        - displaced_outside: Knocked pellet outside

    NON-CAUSAL REACHES:
        - miss_on_pillar: Reach occurred BEFORE interaction, pellet was available (had a chance)
        - miss_off_pillar: Reach occurred AFTER interaction, pellet already gone (no chance)

    SPECIAL CASES:
        - untouched_segment: Segment where pellet was never contacted (all reaches are miss_on_pillar)
"""
import pandas as pd
from pathlib import Path


def derive_reach_outcome(row, is_causal: bool) -> str:
    """
    Derive the individual reach outcome based on timing relative to interaction.

    Args:
        row: DataFrame row with reach data
        is_causal: Whether THIS reach is the causal reach (determined by caller)

    Logic:
        1. If this is the causal reach → use segment outcome (retrieved/displaced_*)
        2. If segment outcome is 'untouched' → all reaches are miss_on_pillar
        3. If reach ends BEFORE interaction_frame → miss_on_pillar (had a chance)
        4. If reach starts AFTER interaction_frame → miss_off_pillar (no chance)
    """
    # Causal reach gets the segment outcome
    if is_causal:
        return row['outcome']  # retrieved, displaced_sa, displaced_outside

    # If segment was untouched, all reaches are misses with the pellet available
    if row['outcome'] == 'untouched':
        return 'miss_on_pillar'

    # Non-causal reach: determine timing relative to interaction
    interaction = row.get('interaction_frame')

    # If no interaction frame recorded, can't determine timing
    if pd.isna(interaction):
        return 'miss_unknown'

    reach_end = row['end_frame']
    reach_start = row['start_frame']

    # Reach ended before interaction → pellet was still on pillar → had a chance
    if reach_end < interaction:
        return 'miss_on_pillar'

    # Reach started after interaction → pellet was already displaced/retrieved → no chance
    if reach_start > interaction:
        return 'miss_off_pillar'

    # Reach overlaps interaction but isn't causal → treat as on_pillar (edge case)
    return 'miss_on_pillar'


def find_causal_reach(segment_df) -> int:
    """
    Find which reach in a segment is the causal reach based on interaction_frame timing.

    Returns the reach_num_in_segment of the causal reach, or -1 if none found.
    """
    interaction = segment_df['interaction_frame'].iloc[0]

    # If no interaction, no causal reach
    if pd.isna(interaction):
        return -1

    # Find reach that contains the interaction frame
    for _, row in segment_df.iterrows():
        if row['start_frame'] <= interaction <= row['end_frame']:
            return row['reach_num_in_segment']

    # No reach contains interaction frame - find closest reach ending before interaction
    # This handles cases where interaction happens between reaches
    before_interaction = segment_df[segment_df['end_frame'] <= interaction]
    if len(before_interaction) > 0:
        # Return the last reach before interaction
        return before_interaction.iloc[-1]['reach_num_in_segment']

    return -1


def main():
    pipeline_dir = Path(__file__).parent

    # Load data
    print('Loading reach data...')
    df = pd.read_excel(pipeline_dir / 'unified_reaches_for_presentation.xlsx', sheet_name='All_Reaches')
    print(f'  {len(df)} reaches from {df["animal"].nunique()} mice')

    # Show current outcome distribution (segment-level)
    print('\nCurrent outcome distribution (segment-level):')
    for outcome, count in df['outcome'].value_counts().items():
        print(f'  {outcome}: {count} ({count/len(df)*100:.1f}%)')

    # Derive individual reach outcomes
    print('\nDeriving individual reach outcomes...')
    df['reach_outcome'] = df.apply(derive_reach_outcome, axis=1)

    # Show new reach outcome distribution
    print('\nDerived reach outcome distribution:')
    for outcome, count in df['reach_outcome'].value_counts().items():
        pct = count / len(df) * 100
        print(f'  {outcome}: {count} ({pct:.1f}%)')

    # Sanity checks
    print('\nSanity checks:')

    # Causal reaches should match segment count
    causal_count = df['is_causal_reach'].sum()
    segment_count = df.groupby(['video_name', 'segment_num']).ngroups
    print(f'  Causal reaches: {causal_count}')
    print(f'  Segments with causal reach: {segment_count}')

    # Retrieved should equal causal reaches in retrieved segments
    retrieved_reaches = (df['reach_outcome'] == 'retrieved').sum()
    print(f'  Retrieved reaches: {retrieved_reaches}')

    # Show example segment
    print('\nExample segment breakdown:')
    example = df[df['video_name'] == df['video_name'].iloc[0]]
    seg1 = example[example['segment_num'] == 1]
    print(f'  Video: {seg1["video_name"].iloc[0]}, Segment 1:')
    print(f'  Segment outcome: {seg1["outcome"].iloc[0]}')
    print(f'  Interaction frame: {seg1["interaction_frame"].iloc[0]}')
    for _, r in seg1.iterrows():
        print(f'    Reach {r["reach_num_in_segment"]}: frames {r["start_frame"]}-{r["end_frame"]} -> {r["reach_outcome"]} (causal={r["is_causal_reach"]})')

    # Save updated data
    output_path = pipeline_dir / 'reaches_with_individual_outcomes.xlsx'
    df.to_excel(output_path, index=False)
    print(f'\nSaved: {output_path}')

    return df


if __name__ == '__main__':
    df = main()
