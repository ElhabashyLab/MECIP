


def give_info_on_params(params, out_file=None):
        # prints or creates txt file with all information on the used params for this run
        out = str(params)
        if out_file:
            with open(out_file, 'w') as f:
                f.write(out)
        else:
            # print summed info
            print(out)