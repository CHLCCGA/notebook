# Git

[Git toturial](bilibili.com/video/BV1db4y1d79C/?spm_id_from=333.1007.top_right_bar_window_custom_collection.content.click&vd_source=40cfaaee5267dcb4afa3c72ce07b463d)

---

+ 克隆仓库 `git clone <git地址>`
+ 版本`git --version`
+ 初始化仓库`git init`



+ 添加文件到暂存区`git add -A`
+ 把暂存区的文件提交到仓库`git commit -m "提交信息"`
+ 查看提交的历史记录`git log --stat`



+ 工作区回滚`git checkout <filename>`  *README.md*
+ 撤销最后一次提交`git reset HEAD^1`  *重置 HEAD 当前提交 ^1:上一次提交*



+ 以当前分支为基础新建分支`git checkout -b <branchname>`
+ 列举所有的分支`git branch`
+ 单纯地切换到某个分支`git checkout <branchname>` 
+ 删掉特定的分支`git branch -D <branchname>`
+ 合并分支`git merge <branchname>`  *在master 或一个分支里 输入要merge的分支*
+ 放弃合并分支(有冲突)`git merge --abort`



+ 推送当前分支最新的提交到远程`git push`
+ 拉取远程分支最新的提交到本地`git pull`