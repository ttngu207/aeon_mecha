import os
import glob
import bisect
import datetime
import pandas as pd
from pathlib import Path

"""The duration of each acquisition chunk, in whole hours."""
CHUNK_DURATION = 1

def aeon(seconds):
    """Converts a Harp timestamp, in seconds, to a datetime object."""
    return datetime.datetime(1904, 1, 1) + pd.to_timedelta(seconds, 's')

def chunk(time):
    '''
    Returns the whole hour acquisition chunk for a measurement timestamp.
    
    :param datetime or Series time: An object or series specifying the measurement timestamps.
    :return: A datetime object or series specifying the acquisition chunk for the measurement timestamp.
    '''
    if isinstance(time, pd.Series):
        hour = CHUNK_DURATION * (time.dt.hour // CHUNK_DURATION)
        return pd.to_datetime(time.dt.date) + pd.to_timedelta(hour, 'h')
    else:
        hour = CHUNK_DURATION * (time.hour // CHUNK_DURATION)
        return pd.to_datetime(datetime.datetime.combine(time.date(), datetime.time(hour=hour)))

def chunk_range(start, end):
    '''
    Returns a range of whole hour acquisition chunks.

    :param datetime start: The left bound of the time range.
    :param datetime end: The right bound of the time range.
    :return: A DatetimeIndex representing the acquisition chunk range.
    '''
    return pd.date_range(chunk(start), chunk(end), freq=pd.DateOffset(hours=CHUNK_DURATION))

def chunk_key(file):
    """Returns the acquisition chunk key for the specified file name."""
    filename = os.path.split(file)[-1]
    filename = os.path.splitext(filename)[0]
    chunk_str = filename.split("_")[-1]
    date_str, time_str = chunk_str.split("T")
    return datetime.datetime.fromisoformat(date_str + "T" + time_str.replace("-", ":"))

def chunk_filter(files, timefilter):
    '''
    Filters a list of paths using the specified time filter. To use the time filter, files
    must conform to a naming convention where the timestamp of each acquisition chunk is
    appended to the end of each file path.

    :param str files: The list of acquisition chunk files to filter.
    :param iterable or callable timefilter:
    A list of acquisition chunks or a predicate used to test each file time.
    :return: A list of all matching filenames.
    '''
    try:
        chunks = [chunk for chunk in iter(timefilter)]
        timefilter = lambda x:x in chunks
    except TypeError:
        if not callable(timefilter):
            raise TypeError("timefilter must be iterable or callable")

    matches = []
    for file in files:
        chunk = chunk_key(file)
        if not timefilter(chunk):
            continue
        matches.append(file)
    return matches

def _set_index(data):
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = aeon(data.index)
    data.index.name = 'time'

def _empty(columns):
    return pd.DataFrame(columns=columns, index=pd.DatetimeIndex([], name='time'))

def load(root, reader, start=None, end=None, time=None, tolerance=None):
    '''
    Extracts chunk data from the root path of an Aeon dataset using the specified data stream
    reader. A subset of the data can be loaded by specifying an optional time range, or a list
    of timestamps used to index the data on file. Returned data will be sorted chronologically.

    :param str root: The root path, or prioritised sequence of paths, where epoch data is stored.
    :param Reader reader: A data stream reader object used to read chunk data from the dataset.
    :param datetime, optional start: The left bound of the time range to extract.
    :param datetime, optional end: The right bound of the time range to extract.
    :param datetime, optional time: An object or series specifying the timestamps to extract.
    :param datetime, optional tolerance:
    The maximum distance between original and new timestamps for inexact matches.
    :return: A pandas data frame containing epoch event metadata, sorted by time.
    '''
    files = set()
    if isinstance(root, str):
        root = [root]
    for path in root:
        files.update(Path(path).glob(f"**/**/{reader.name}*.{reader.extension}"))
    files = sorted(files)

    if time is not None:
        # ensure input is converted to timestamp series
        if isinstance(time, pd.DataFrame):
            time = time.index
        if not isinstance(time, pd.Series):
            time = pd.Series(time)
            time.index = time

        dataframes = []
        filetimes = [chunk_key(file) for file in files]
        for key,values in time.groupby(by=chunk):
            i = bisect.bisect_left(filetimes, key)
            if i < len(filetimes):
                frame = reader.read(files[i])
                _set_index(frame)
            else:
                frame = _empty(reader.columns)
            data = frame.reset_index()
            data.set_index('time', drop=False, inplace=True)
            data = data.reindex(values, method='pad', tolerance=tolerance)
            missing = len(data.time) - data.time.count()
            if missing > 0 and i > 0:
                # expand reindex to allow adjacent chunks
                # to fill missing values
                previous = reader.read(files[i-1])
                data = pd.concat([previous, frame])
                data = data.reindex(values, method='pad', tolerance=tolerance)
            else:
                data.drop(columns='time', inplace=True)
            dataframes.append(data)

        if len(dataframes) == 0:
            return _empty(reader.columns)
            
        return pd.concat(dataframes)

    if start is not None or end is not None:
        timefilter = chunk_range(start, end)
        files = chunk_filter(files, timefilter)
    else:
        timefilter = None

    if len(files) == 0:
        return _empty(reader.columns)

    data = pd.concat([reader.read(file) for file in files])
    _set_index(data)
    if timefilter is not None:
        try:
            return data.loc[start:end]
        except KeyError:
            import warnings
            if not data.index.has_duplicates:
                warnings.warn('data index for {0} contains out-of-order timestamps!'.format(reader.name))
                data = data.sort_index()
            else:
                warnings.warn('data index for {0} contains duplicate keys!'.format(reader.name))
                data = data[~data.index.duplicated(keep='first')]
            return data.loc[start:end]
    return data