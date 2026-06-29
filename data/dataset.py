
import argparse
import sys
import casanova


def main():
    parser = argparse.ArgumentParser(
        description="Concatenate multiple CSV files into a single dataframe"
    )
    parser.add_argument("files", nargs="+", help="Input CSV files to concatenate")
    parser.add_argument("--chunks", default=False, action="store_true", help="Whether to divide speeches in chunks")
    parser.add_argument("--no-vote", default=False, action="store_true", help="Whether to generate a dataset for interventions without any vote")

    args = parser.parse_args()

    concat = not args.chunks

    header = [
        "speech_id",
        "debate_id",
        "speech_date",
        "speech_subject_1",
        "speech_subject_2",
        "speech_subject_3",
        "speaker_name",
        "speaker_group",
        "speaker_government",
        "vote",
        "source",
        "speech",
        "amendments",
        "amendments_precised"
    ]

    rows = []
    for file in args.files:
        with casanova.reader(file) as reader:
            for row in reader:
                rows.append(row)

    

    with casanova.writer(sys.stdout, header) as writer:
        (
            t_speech_id,
            t_debate_id,
            t_speech_date,
            t_subject,
            t_s_subject,
            t_ss_subject,
            t_speaker_name,
            t_speaker_group,
            t_speaker_government,
            t_label,
            t_type,
            t_speech,
            t_amd,
            t_amd_p
        ) = None, None, None, None, None, None, None, None, None, None, None, [], None, None
        has_current = False
        results = []
        for row in rows:
            (
                intervention_id,
                debate_number,
                debate_ref,
                date,
                moment,
                subject,
                sub_subject,
                name,
                sexe,
                group,
                last_group,
                function,
                code,
                intervention,
                sub_sub_subject,
                amendments,
                amendments_precise,
                is_government,
                vote_id,
                vote_issue,
                vote_type
            ) = row

            if not args.no_vote and not vote_issue:
                continue
            if not name:
                continue

            if code != "PAROLE_GENERIQUE":
                continue
            if function == "président":
                continue

            if "appel au règlement" in sub_sub_subject:
                continue
            if "appels au règlement" in sub_sub_subject:
                continue

            if not args.no_vote:
                vote_id = vote_id.split('|')[0]

                vote_parts = set(vote_issue.split("|"))

                if "POUR" in vote_parts and "CONTRE" in vote_parts:
                    continue
                if "POUR" in vote_parts:
                    vote_issue = "POUR"
                elif "CONTRE" in vote_parts:
                    vote_issue = "CONTRE"
                else:
                    continue
            else:
                vote_issue = None

            speech_id = f"{vote_id}-{intervention_id}"

            if concat:
                if (t_debate_id, t_speech_date, t_subject, t_s_subject, t_ss_subject, t_speaker_name, t_speaker_group, t_speaker_government) != (vote_id, date, subject, sub_subject, sub_sub_subject, name, group, is_government):
                    if has_current:
                        final_speech = ' '.join(t_speech)

                        results.append([
                            t_speech_id,
                            t_debate_id,
                            t_speech_date,
                            t_subject,
                            t_s_subject,
                            t_ss_subject,
                            t_speaker_name,
                            t_speaker_group,
                            t_speaker_government,
                            t_label,
                            t_type,
                            final_speech,
                            t_amd,
                            t_amd_p
                        ])

                    (
                        t_speech_id,
                        t_debate_id,
                        t_speech_date,
                        t_subject,
                        t_s_subject,
                        t_ss_subject,
                        t_speaker_name,
                        t_speaker_group,
                        t_speaker_government,
                        t_label,
                        t_type,
                        t_speech,
                        t_amd,
                        t_amd_p
                    ) = speech_id, vote_id, date, subject, sub_subject, sub_sub_subject, name, group, is_government, vote_issue, vote_type, [], amendments, amendments_precise
                    has_current = True

                    if intervention:
                        t_speech.append(intervention)
                else:
                    if not t_speech or t_speech[-1] != intervention:
                        t_speech.append(intervention)
                    
            else:
                results.append([
                    speech_id,
                    vote_id,
                    date,
                    subject,
                    sub_subject,
                    sub_sub_subject,
                    name,
                    group,
                    is_government,
                    vote_issue,
                    vote_type,
                    intervention,
                    amendments,
                    amendments_precise
                ])

        if concat and has_current:
            final_speech = ' '.join(t_speech)
            results.append([
                t_speech_id,
                t_debate_id,
                t_speech_date,
                t_subject,
                t_s_subject,
                t_ss_subject,
                t_speaker_name,
                t_speaker_group,
                t_speaker_government,
                t_label,
                t_type,
                final_speech,
                t_amd,
                t_amd_p
            ])
            
        writer.writerows(results)



if __name__ == "__main__":
    main()
