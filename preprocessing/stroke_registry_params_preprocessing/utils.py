import pandas as pd


def set_sample_date(stroke_registry_df):
    """
    Set sample date to stroke onset or arrival at hospital, whichever is later
    """
    datatime_format = '%d.%m.%Y %H:%M'
    stroke_registry_df['arrival_dt'] = pd.to_datetime(stroke_registry_df['Arrival at hospital'],
                                                      format='%Y%m%d').dt.strftime('%d.%m.%Y') + ' ' + \
                                       pd.to_datetime(stroke_registry_df['Arrival time'], format='%H:%M',
                                                      infer_datetime_format=True).dt.strftime('%H:%M')

    stroke_registry_df['stroke_dt'] = pd.to_datetime(stroke_registry_df['Onset date'],
                                                     format='%Y%m%d').dt.strftime('%d.%m.%Y') + ' ' + \
                                        pd.to_datetime(stroke_registry_df['Onset time'], format='%H:%M',
                                                       infer_datetime_format=True).dt.strftime('%H:%M')

    # set sample date to stroke onset or arrival at hospital, whichever is later
    # this takes into account potential in-hospital stroke events
    stroke_registry_df['delta_onset_arrival'] = (
            pd.to_datetime(stroke_registry_df['stroke_dt'], format=datatime_format, errors='coerce')
            - pd.to_datetime(stroke_registry_df['arrival_dt'], format=datatime_format, errors='coerce')
    ).dt.total_seconds()
    stroke_registry_df['sample_date'] = stroke_registry_df \
        .apply(lambda x: x['stroke_dt'] if x['delta_onset_arrival'] > 0 else x['arrival_dt'],
               axis=1)

    return stroke_registry_df