from scipy.stats import f_oneway

def anova_by_group(df, group_col, value_col):
    groups = [sub[value_col].dropna() for _,sub in df.groupby(group_col)]
    f, p = f_oneway(*groups)
    return f, p

