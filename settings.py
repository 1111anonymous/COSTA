import datetime

task_name = str(datetime.datetime.now().date())
city = 'SIN'  # PHO, NYC, SIN
gpuId = "cuda:0"

enable_random_mask = True
mask_prop = 0.1
cl_sample = 10
cl_loss = 'log' # log or bpr
extra_input = 'none' # 'none': transformer input with l,c,u,t,d; / 'pop': transformer input add pop seq; / 'dist': transformer input add dist seq; / 'all': transformer input add pop & dist seq
cl_enhance = 'all' # 'none': loc embedding; / 'pop': loc + pop embedding; / 'dist': loc + dist embedding; / 'all': loc + pop + dist embedding
info_enhance = 'all' # 'user' / 'pop' / 'dist' / 'all' / 'none'
sample_feature = 'all' # 'pop' / 'dist' / 'all': rank according dist at first, select the top50, then rank according to pop



if city == 'SIN':
    lr = 1e-4
    embed_size = 90
    cl_weight = 3
    epoch = 50
    run_times = 3
elif city == 'NYC':
    lr = 1e-3
    embed_size = 60
    cl_weight = 4
    epoch = 50
    run_times = 3
elif city == 'PHO':
    lr = 1e-4
    embed_size = 120
    cl_weight = 3
    epoch = 50#100
    run_times = 5
elif city == 'CAL':
    lr = 1e-4
    embed_size = 120
    cl_weight = 3
    epoch = 100
    run_times = 5

output_file_name = f'{city}' + '_ClLoss' + str(cl_loss) + '_Sample' + str(sample_feature) + '_TransInput' + str(extra_input) + '_ClEnhance' + str(cl_enhance) + '_infoEnhance' + str(info_enhance)

output_file_name = output_file_name + '_ClWeight' + str(cl_weight) + '_embeddingSize' + str(embed_size) + '_lr' + str(lr) + '_epoch' + str(epoch)