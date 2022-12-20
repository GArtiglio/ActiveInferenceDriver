library(car)
library(tidyverse)
library(data.table)
library(parsnip)
library(tidymodels)
library(skimr)
library(vip)
library(knitr)
library(themis)
library(ggforce)

theme_set(theme_bw(base_size=18, base_family="Time New Roman"))
some <- function(dat){car::some(dat)}

cwd = dirname(getwd())
exp_path = file.path(
  cwd, "exp/ebirl", 
  "lognormal_state_dim_2_embedding_dim_None_reward_ahead_True_prior_dist_mvn_num_components_3_post_dist_diag_iterative_True_init_uniform_True_obs_penalty_0.2_prior_lr_0.01_vi_lr_0.01_decay_0_seed_0"
)
fig_path = file.path(exp_path, "fig")
dir.create(fig_path, showWarnings=FALSE)

## Plot counterfactual violin
count_fact = fread(file.path(exp_path, "tab", "cf_simulation.csv"))
some(count_fact)
count_fact = mutate(
  count_fact, 
  Response = ifelse(
    RT < Delay, "Driver braked before AEB", 
    ifelse(RT<4.90, "Driver braked after AEB", "Driver did not brake"))
)
count_fact = mutate(
  count_fact, 
  Delay = as.factor(as.character(Delay))) %>% filter(Delay != "0"
  )
count_fact = count_fact[count_fact$Delay %in% c(0.5, 1.5, 2.5, 3.5, 4.5),]

p = ggplot(
  count_fact, aes(x=Delay, y=RT)) + 
  geom_violin() +
  geom_sina(aes(group=Delay, color=Response), size=1.5) + 
  xlab("AEB delay (s)") +
  ylab("Time-to-decision (s)") +
  scale_color_manual(values=c("#5380bc","#500000","grey55")) +
  theme(
    text=element_text(family="Times New Roman"), 
    legend.position="top", 
    legend.title=element_blank()
  )

ggsave(
  file.path(fig_path, "counterfactual_violin.png"), 
  width=8, 
  height=4, 
  units="in", 
  p
)

# ## Plot counter factual factor 4
# some(count_fact)
# ggplot(count_fact, aes(x=Factor_4, y=RT, color=Response))+geom_point(size=2)+xlab("Factor 4")+ylab("Reaction time (s)")+facet_grid(Delay~.)+
#   scale_color_manual(values = c("#5380bc","#500000","grey55"))+theme(legend.position = "top",legend.title=element_blank())
# 
# ggsave(file.path(fig_path, "cf_fact4.png"),  width=10.62, height = 6, units="in",
#        ggplot(count_fact, aes(x=Factor_4, y=RT, color=Response))+geom_point(size=2)+xlab("Factor 4")+ylab("Reaction time (s)")+facet_grid(Delay~.)+
#          scale_color_manual(values = c("#5380bc","#500000","grey55"))+theme(legend.position = "top",legend.title=element_blank()))
# 
# ## Plot prior and posterior predictive checks
# prpr_check = fread(file.path(exp_path, "tab", "sample_predictive.csv"))
# some(prpr_check)
# prpr_check = prpr_check %>% mutate(rt = round(rt/10,1))
# ggplot(prpr_check, aes(x=rt, group = Distribution, color=Distribution))+stat_ecdf(geom="step", size=1) + facet_wrap(~Scenario)+
#   scale_color_manual(values = c("#5380bc","black","grey55"))+ylab("Cumulative density")+xlab("Reaction time (s)")
# 
# ggsave(file.path(fig_path, "cumulative_dens.png"), width=10.62, height = 6, units="in",
#        ggplot(prpr_check, aes(x=rt, group = Distribution, color=Distribution))+stat_ecdf(geom="step", size=1) + facet_wrap(~Scenario)+
#   scale_color_manual(values = c("#5380bc","black","grey55"))+ylab("Cumulative density")+xlab("Reaction time (s)"))
# 
# ## Plot factor 4
# fact_2 = fread(file.path(exp_path, "tab", "factor_assignment.csv"))
# some(fact_2)
# fact_2 = fact_2 %>% mutate(Rt = Rt/10)
# ggplot(fact_2, aes(x=Factor_4, y = Rt))+geom_point(size = 2)+geom_smooth(method="glm", color="#500000")+ylab("Reaction Time (s)")+xlab("Factor 4")
# 
# ggsave(file.path(fig_path, "Factor_4_cor.png"), width=10.62, height = 6, units="in",
# ggplot(fact_2, aes(x=Factor_4, y = Rt))+geom_point(size = 4)+geom_smooth(method="glm", color="#500000")+ylab("Reaction Time (s)")+xlab("Factor 4"))