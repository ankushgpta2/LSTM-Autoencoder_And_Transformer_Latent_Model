from dnn_modules import *
import os
import scipy.io
import h5py
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler, KBinsDiscretizer, PolynomialFeatures, StandardScaler, RobustScaler, MinMaxScaler


def analyze_tiago_data_main(general_parameters, autoencoder_parameters):
    """
        THIS FUNCTION IS THE MAIN ONE FOR ACTUALLY ACQUIRING THE DATA FROM THE SPECIFIED DIRECTORIES AND STORING IN PROPER PLACE + CLEANING IT... CALLS THE FEW FUNCTIONS BELOW THIS ONE
    """
    name_of_experiment, data_for_different_recording_periods = get_all_tiago_data(general_parameters)
    run_model(general_parameters, autoencoder_parameters, input_data=data_for_different_recording_periods[general_parameters['dataset']][0],
    input_labels=data_for_different_recording_periods[general_parameters['dataset']][1], experiment_name=name_of_experiment)


def get_all_tiago_data(general_parameters):
    # load data
    all_data_from_matlab, mat_paths = get_data(data_directory=general_parameters['data_directory'])
    name_of_experiment = mat_paths[general_parameters['exp_num']]
    name_of_experiment = name_of_experiment.split('/')[-1]
    name_of_experiment = name_of_experiment.split('.')[0]
    specific_exp = all_data_from_matlab[general_parameters['exp_num']]
    cleaned_full_df = clean_the_data(data=specific_exp,
                                     threshold_percentile=general_parameters['threshold_percentile'],
                                     threshold_for_minimum_activity=general_parameters['threshold_for_minimum_activity'],
                                     type_of_scaling=general_parameters['scaling'],
                                     drop_cells=general_parameters['drop_cells'])
    full_x_data = cleaned_full_df[cleaned_full_df.columns[:-2]].to_numpy()
    full_trial_data = cleaned_full_df[cleaned_full_df.columns[-2]].to_numpy()
    full_on_data = cleaned_full_df[cleaned_full_df.columns[-1]].to_numpy()
    on, y_data_for_on, hundred, five_hundred, y_data_for_100, y_data_for_500 = separate_recording_periods(full_x_data, full_on_data, full_trial_data)
    data_for_different_recording_periods = {'Full': [full_x_data, full_trial_data, full_on_data], 'On': [on, y_data_for_on], 'Hundred': [hundred, y_data_for_100], 'Five Hundred': [five_hundred, y_data_for_500]}
    return name_of_experiment, data_for_different_recording_periods


def get_data(data_directory):
    mat_paths = []
    txt_paths = []
    for x in range(len(data_directory)):
        data_path = data_directory[x]
        directory_list_mat, directory_list_orientations = get_names_of_files(data_path)
        mat_paths.append(directory_list_mat)
        txt_paths.append(directory_list_orientations)

    struct_4_data = create_class_4_data()
    meta_data_struct = meta_data()

    all_data_from_matlab = []
    number_of_iterations, number_of_iterations2 = np.shape(mat_paths)
    txt_paths = np.asarray(txt_paths).flatten()
    for i in range(number_of_iterations):
        for x in range(number_of_iterations2):
            try:
                matlab_data = h5py.File(mat_paths[i][x])
            except:
                matlab_data = scipy.io.loadmat(mat_paths[i][x])
            text_file_name = txt_paths[i]
            trials = (np.loadtxt(text_file_name, dtype=int))
            all_data_from_matlab = get_all_data(matlab_data, trials, struct_4_data, all_data_from_matlab)
            matlab_data.close()
    return all_data_from_matlab, np.asarray(mat_paths).flatten()


def get_names_of_files(data_directory):
    extension = r".mat"
    extension2 = r".txt"
    directory_list_mat = [os.path.join(data_directory, _) for _ in os.listdir(data_directory) if _.endswith(extension)]
    directory_list_orientations = [os.path.join(data_directory, _) for _ in os.listdir(data_directory) if
                                   _.endswith(extension2)]
    return directory_list_mat, directory_list_orientations


def meta_data():
    class meta_data_struct(object):
        def __init__(self, mouse_id, recording_date, day):
            self.mouse_id = mouse_id
            self.recording_date = recording_date
            self.day = day
    return meta_data_struct


def create_class_4_data():
    class struct_4_data(object):
        def __init__(self, YCsignal, amp, basedrift, calciumfit, decay, framerate, sigma, spikes, visualstimulus, orientation_trials):
            self.YCsignal = YCsignal
            self.amp = amp
            self.basedrift = basedrift
            self.calciumfit = calciumfit
            self.decay = decay
            self.framerate = framerate
            self.sigma = sigma
            self.spikes = spikes
            self.visualstimulus = visualstimulus
            self.orientation_trials = orientation_trials
    return struct_4_data


def get_all_data(matlab_data, trials, struct_4_data, all_data_from_matlab):
    YCsignal = np.array(matlab_data['YCsignal'])
    amp = np.array(matlab_data['amp'])
    basedrift = np.array(matlab_data['basedrift'])
    calciumfit = np.array(matlab_data['calciumfit'])
    decay = np.array(matlab_data['decay'])
    framerate = np.float32(matlab_data['framerate'])
    sigma = np.array(matlab_data['sigma'])
    spikes = np.array(matlab_data['spikes'])
    visualstimulus = np.array(matlab_data['visualstimulus'])
    orientation_trials = trials
    all_data_from_matlab.append(struct_4_data(YCsignal, amp, basedrift, calciumfit, decay, framerate, sigma, spikes, visualstimulus, orientation_trials))
    return all_data_from_matlab


def get_meta_data(meta_data_struct, directory_list_orientations):
    all_meta_data = []
    for name in directory_list_orientations:
        name = name.split('/')[-1]
        exp_meta_info = re.findall(r'\d+', str(name))
        if len(exp_meta_info) == 4:
            case = 1
        elif len(exp_meta_info) == 5:
            case = 2
        else:
            sys.exit('Unable to Resolve Meta Data for Recording')
        mouse_id = str(exp_meta_info[0])
        recording_date = str(exp_meta_info[1] + '.' + exp_meta_info[2] + '.' + exp_meta_info[3])
        if case == 1:
            day = ''
        else:
            day = exp_meta_info[4]
        all_meta_data.append(meta_data_struct(mouse_id, recording_date, day))
    return all_meta_data


def organize_data(all_meta_data, directory_list_mat, directory_list_orientations):
    mouse_id_list = []
    day_list = []
    for i in range(len(all_meta_data)):
        mouse_id_list.append(all_meta_data[i].mouse_id)
        day_list.append(all_meta_data[i].day)

    unique_id_list = list(set(mouse_id_list))
    indices_list = []
    for i in unique_id_list:
        if mouse_id_list.count(i) > 1:
            for ii in range(len(mouse_id_list)):
                if i == mouse_id_list[ii]:
                    indices_list.append(ii)

    for elem in indices_list:
        if day_list[elem] == '':
            day_list[elem] = '1'
    text_file_name_4_orientations = align_meta_and_mat(mouse_id_list, day_list, all_meta_data, directory_list_mat,
                                                       directory_list_orientations)
    return text_file_name_4_orientations


def align_meta_and_mat(mouse_id_list, day_list, all_meta_data, directory_list_mat, directory_list_orientations):
    num_match = []
    text_file_name_4_orientations = []
    for element in directory_list_mat:
        name_of_mat_file = str(element)
        relevant_part_of_mat = re.findall(r'\d+', str(name_of_mat_file.split('/')[-1]))
        if len(relevant_part_of_mat) == 1:
            relevant_part_of_mat.append('')
        for ii in range(len(all_meta_data)):
            [exp_mouse_id, exp_day] = str(mouse_id_list[ii]), str(day_list[ii])
            check1 = int(relevant_part_of_mat[0] == exp_mouse_id[-4:])
            check2 = int(relevant_part_of_mat[1] == exp_day)
            num_match.append(check1 + check2)
        text_file_name_4_orientations.append(str(directory_list_orientations[num_match.index(max(num_match))]))
        num_match.clear()
    return text_file_name_4_orientations


def parse_evoked(visualstim_data, trial_info):
    stimulus_frames = visualstim_data
    trial_counter = 0
    trial_frames = np.zeros([1000, 4])  # just create an array that is 1000 x 3 filled with zeros
    num_of_frames = np.shape(stimulus_frames)[1]
    i = 0
    while i <= num_of_frames - 1:
        if stimulus_frames[0, i] == 1:
            trial_counter += 1
            trial_start = i
            while stimulus_frames[0, i] == 1:
                i += 1
            on_for_trial_end = i
            while stimulus_frames[0, i] == 0 and i != num_of_frames-1:
                i += 1
            trial_end = i
            trial_frames[trial_counter, 0] = trial_start
            trial_frames[trial_counter, 1] = on_for_trial_end
            trial_frames[trial_counter, 2] = trial_end
        else:
            i += 1
    total_num_of_trials = trial_counter
    for i in range(len(trial_info)):
        trial_frames[i, 3] = int(trial_info[i])
    trial_frames = trial_frames[~np.all(trial_frames == 0, axis=1)]
    return trial_frames, total_num_of_trials


def convert_to_dataframe(data, y_train_info, num_of_trials):
    [num_of_frames, num_of_neurons] = np.shape(data)
    columns = ['ROI_' + str(i) for i in range(1, num_of_neurons+1)]
    df_spikes = pd.DataFrame(data, columns=columns)
    # do some cleaning up
    df_spikes = df_spikes.replace([np.inf, -np.inf], np.nan)  # convert inf to nan
    df_spikes.fillna(df_spikes.mean(), inplace=True)
    full_df, list_for_grating_column = add_grating_labels_to_df(df_spikes, y_train_info, num_of_trials, num_of_frames)
    return full_df, y_train_info


def add_grating_labels_to_df(df_spikes, y_train_info, num_of_trials, num_of_frames):
    list_for_trial = np.empty((num_of_frames, 1))
    list_for_on_periods = np.empty((num_of_frames, 1))
    list_for_trial[:] = np.nan
    list_for_on_periods[:] = np.nan
    for i in range(num_of_trials):
        list_for_trial[np.int(y_train_info[i, 0]):np.int(y_train_info[i, 2])] = y_train_info[i, 3]
        list_for_on_periods[np.int(y_train_info[i, 0]):np.int(y_train_info[i, 1])] = y_train_info[i, 3]
    list_for_on_periods[np.isnan(list_for_on_periods)] = 8
    df_spikes['orientation_class'] = list_for_trial
    df_spikes['orientation_class_on'] = list_for_on_periods
    # drop frames until first trial and drop frames after last trial
    df_spikes = df_spikes.dropna(subset=['orientation_class'])
    full_df = df_spikes
    return full_df, y_train_info


def clean_the_data(data, threshold_percentile, threshold_for_minimum_activity, type_of_scaling, drop_cells):
    # add column to dataframe containing the orientation grating for each frame (0-7)... 8 = non stim presentation frames
    trial_frames, total_num_of_trials = parse_evoked(visualstim_data=data.visualstimulus,
                                                     trial_info=data.orientation_trials)
    full_df, y_train_info = convert_to_dataframe(num_of_trials=total_num_of_trials, data=data.spikes,
                                                 y_train_info=trial_frames)  # --> convert to df for convenience

    # get a better understanding of data and clean up a little bit
    threshold_data = threshold_spiking_prob(data=full_df,
                                            threshold_percentile=threshold_percentile)
    # thresholds based on percentile of distribution
    avg_activity_per_roi, cleaned_data = explore_spiking(data=threshold_data, threshold=threshold_for_minimum_activity, drop_cells=drop_cells)

    # split the cleaned up data into train and test or train, validate, and test (based on # of inputs)
    cleaned_full_df = scale_the_data(cleaned_data, type_of_scaling=type_of_scaling)

    return cleaned_full_df


def threshold_spiking_prob(data, threshold_percentile):
    convert_to_numpy = data.to_numpy()
    lowest_percent = np.percentile(convert_to_numpy, threshold_percentile)
    convert_to_numpy[convert_to_numpy < lowest_percent] = 0
    threshold_data = data
    return threshold_data


def explore_spiking(data, threshold, drop_cells):
    avg_activity_per_roi = data.mean(axis=0)
    cleaned_df = data.replace([np.inf, -np.inf], np.nan)  # convert inf to nan
    cleaned_df = cleaned_df.fillna(0)  # replace all nans as 0s
    if drop_cells is True:
        cleaned_data = drop_inactive_cells(avg_activity_data=avg_activity_per_roi, threshold=threshold, data_to_be_cleaned=cleaned_df)
    else:
        cleaned_data = cleaned_df
    return avg_activity_per_roi, cleaned_data


def drop_inactive_cells(threshold, data_to_be_cleaned, avg_activity_data):
    indices = [i for i, v in enumerate(avg_activity_data.to_numpy()) if v < threshold]
    keys_for_indices = [str('ROI_' + str(i+1)) for i in indices]
    data_to_be_cleaned.drop(keys_for_indices, inplace=True, axis=1)
    cleaned_data = data_to_be_cleaned
    return cleaned_data


def scale_the_data(cleaned_data, type_of_scaling):
    # scale entire dataframe
    if type_of_scaling == 'MaxAbsScaler':
        scaler = MaxAbsScaler()
    elif type_of_scaling == 'StandardScaler':
        scaler = StandardScaler()
    elif type_of_scaling == 'MinMaxScaler':
        scaler = MinMaxScaler()
    outliers_removed = RobustScaler().fit_transform(cleaned_data[cleaned_data.columns[:-2]])
    cleaned_data[cleaned_data.columns[:-2]] = scaler.fit_transform(outliers_removed)
    cleaned_data[cleaned_data.columns[:-2]] = cleaned_data[cleaned_data.columns[:-2]]
    return cleaned_data


def separate_recording_periods(full_x_data, full_on_data, full_trial_data):
    # only for stimulus presentation frames
    indices2 = full_on_data != 8
    on = full_x_data[indices2, :]
    y_data_for_on = full_on_data[indices2]
    # first 100 ms or first 5 frames roughly (assuming 45.5 hz imaging rate) and first 500ms or 25 frames
    indices = np.insert(arr=np.where(np.diff(full_trial_data[:]))[0] + 1, obj=0, values=0)
    final_frames_for_5 = []
    final_frames_for_20 = []
    for x in range(0, len(indices)):
        final_frames_for_5.append([i for i in range(indices[x], indices[x] + 5)])
        final_frames_for_20.append([i for i in range(indices[x], indices[x] + 25)])
    final_frames_for_5 = (np.asarray(final_frames_for_5, dtype=np.int)).flatten()
    final_frames_for_20 = (np.asarray(final_frames_for_20, dtype=np.int)).flatten()
    hundred = full_x_data[final_frames_for_5, :]
    five_hundred = full_x_data[final_frames_for_20, :]
    y_data_for_100 = full_on_data[final_frames_for_5]
    y_data_for_500 = full_on_data[final_frames_for_20]
    return on, y_data_for_on, hundred, five_hundred, y_data_for_100, y_data_for_500
