-- Calendar is tiny (1,969 rows), so it is simply rebuilt every run.
TRUNCATE staging.calendar;

INSERT INTO staging.calendar (
    d_num, date, wm_yr_wk, weekday, wday, month, year, day_of_month, is_weekend,
    event_name_1, event_type_1, event_name_2, event_type_2,
    event_type_code, is_event, days_to_next_event, snap_ca, snap_tx, snap_wi)
SELECT
    CAST(dn AS smallint),
    date, wm_yr_wk, weekday, wday, month, year,
    CAST(EXTRACT(DAY FROM date) AS smallint),
    CASE WHEN wday IN (1, 2) THEN 1 ELSE 0 END,          -- M5: wday 1 = Saturday, 2 = Sunday
    event_name_1, event_type_1, event_name_2, event_type_2,
    CASE event_type_1 WHEN 'Cultural' THEN 1 WHEN 'National' THEN 2
                      WHEN 'Religious' THEN 3 WHEN 'Sporting' THEN 4 ELSE 0 END,
    CASE WHEN event_name_1 IS NOT NULL THEN 1 ELSE 0 END,
    -- Looking FORWARD in time is normally leakage. It is fine here only because the
    -- event calendar (Christmas, Super Bowl, ...) is published long in advance, so
    -- on the forecast date we genuinely know when the next event is.
    CAST(LEAST(COALESCE(next_event_d - dn, 30), 30) AS smallint),
    COALESCE(snap_ca, 0), COALESCE(snap_tx, 0), COALESCE(snap_wi, 0)
FROM (
    SELECT c.*,
           CAST(substr(c.d, 3) AS integer) AS dn,
           MIN(CASE WHEN c.event_name_1 IS NOT NULL THEN CAST(substr(c.d, 3) AS integer) END)
               OVER (ORDER BY c.date ROWS BETWEEN 1 FOLLOWING AND UNBOUNDED FOLLOWING) AS next_event_d
    FROM raw.calendar c
) x;

ANALYZE staging.calendar;
